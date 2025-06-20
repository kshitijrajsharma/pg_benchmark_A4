"""
pg_benchmark.py

A comprehensive asynchronous benchmarking tool for PostgreSQL queries.

This script provides utilities to:
- Benchmark SQL queries against a PostgreSQL database, measuring execution time and collecting query plans.
- Compare performance before and after optimizations such as index creation.
- Automatically manage database connections via an asyncpg connection pool.
- Clear PostgreSQL and OS-level caches between benchmark runs for accurate measurement.
- Generate detailed and markdown reports summarizing benchmark results and performance improvements.

Configuration:
- Connection details can be provided as a connection string or via environment variables (DATABASE_URL, or POSTGRES_HOST, POSTGRES_DB, etc.).

Example usage:
- Run the script directly to execute a sample benchmark, compare query performance before and after indexing, and generate a markdown report.

Dependencies: asyncpg, Python 3.10+, and a running PostgreSQL instance with the data you wanna test on.

Author: kshitijrajsharma
"""

import asyncio
import hashlib
import logging
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import asyncpg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_connection_string(connection_string: Optional[str] = None) -> str:
    if connection_string:
        logger.info("Using provided connection string")
        return connection_string

    database_url = os.getenv("DATABASE_URL")
    if database_url:
        logger.info("Using DATABASE_URL from environment")
        return database_url

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")

    if database and user and password:
        conn_str = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        logger.info("Built connection string from individual environment variables")
        return conn_str

    missing_vars = []
    if not database:
        missing_vars.append("POSTGRES_DB")
    if not user:
        missing_vars.append("POSTGRES_USER")
    if not password:
        missing_vars.append("POSTGRES_PASSWORD")

    raise ValueError(
        f"No database connection information provided. "
        f"Either provide connection_string parameter or set environment variables. "
        f"Missing environment variables: {', '.join(missing_vars)}. "
        f"Alternatively, set DATABASE_URL with full connection string."
    )


@dataclass
class QueryExecutionResult:
    execution_time_ms: float
    query_plan: Optional[Dict[str, Any]]
    rows_returned: int
    execution_number: int
    cached: bool = False
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class QueryBenchmarkResult:
    query: str
    query_hash: str
    description: str
    execution_results: List[QueryExecutionResult]
    avg_time_ms: float = 0
    median_time_ms: float = 0
    min_time_ms: float = 0
    max_time_ms: float = 0
    std_dev_ms: float = 0
    total_runs: int = 0
    primary_scan_type: str = ""
    estimated_cost: float = 0
    actual_cost: float = 0

    def __post_init__(self):
        if not self.query_hash:
            self.query_hash = hashlib.md5(self.query.encode()).hexdigest()[:8]

    def calculate_metrics(self):
        if not self.execution_results:
            return

        times = [r.execution_time_ms for r in self.execution_results]
        self.total_runs = len(times)
        self.avg_time_ms = statistics.mean(times)
        self.median_time_ms = statistics.median(times)
        self.min_time_ms = min(times)
        self.max_time_ms = max(times)
        self.std_dev_ms = statistics.stdev(times) if len(times) > 1 else 0

        if self.execution_results[0].query_plan:
            plan = self.execution_results[0].query_plan
            self.primary_scan_type = plan.get("Node Type", "Unknown")
            self.estimated_cost = plan.get("Total Cost", 0)
            self.actual_cost = plan.get("Actual Total Time", 0)


class QueryMomento:
    def __init__(self):
        self.results: Dict[str, QueryBenchmarkResult] = {}
        self.execution_order: List[str] = []

    def store_result(self, result: QueryBenchmarkResult):
        key = f"{result.query_hash}_{result.description}"
        self.results[key] = result
        if key not in self.execution_order:
            self.execution_order.append(key)
        logger.info(
            f"Stored result for: {result.description} (hash: {result.query_hash})"
        )

    def get_result(
        self, query_hash: str, description: str = ""
    ) -> Optional[QueryBenchmarkResult]:
        key = f"{query_hash}_{description}"
        return self.results.get(key)

    def get_all_results(self) -> List[QueryBenchmarkResult]:
        return [
            self.results[key] for key in self.execution_order if key in self.results
        ]

    def compare_results(
        self, hash1: str, desc1: str, hash2: str, desc2: str
    ) -> Dict[str, Any]:
        result1 = self.get_result(hash1, desc1)
        result2 = self.get_result(hash2, desc2)

        if not result1 or not result2:
            return {"error": "One or both results not found"}

        improvement = 0
        if result1.avg_time_ms > 0:
            improvement = (
                (result1.avg_time_ms - result2.avg_time_ms) / result1.avg_time_ms
            ) * 100

        return {
            "baseline": {
                "description": result1.description,
                "avg_time_ms": result1.avg_time_ms,
                "scan_type": result1.primary_scan_type,
            },
            "optimized": {
                "description": result2.description,
                "avg_time_ms": result2.avg_time_ms,
                "scan_type": result2.primary_scan_type,
            },
            "improvement_percent": improvement,
            "faster": "optimized" if improvement > 0 else "baseline",
        }


class PostgreSQLBenchmark:
    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = get_connection_string(connection_string)
        self.pool = None
        self.momento = QueryMomento()
        logger.info(
            f"Initialized PostgreSQL Benchmark for user: {os.getenv('USER', 'unknown')}"
        )

    async def __aenter__(self):
        self.pool = await asyncpg.create_pool(
            self.connection_string, min_size=1, max_size=10, command_timeout=300
        )
        logger.info("Database connection pool created")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")

    async def clear_cache(self):
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("DISCARD ALL;")
                await conn.execute("SELECT pg_stat_reset();")
                try:
                    await conn.execute("SELECT pg_reload_conf();")
                except:
                    pass
                logger.info("Cache cleared successfully")
        except Exception as e:
            logger.warning(f"Could not clear cache: {e}")

    async def execute_query_only(self, query: str) -> Any:
        async with self.pool.acquire() as conn:
            try:
                result = await conn.fetch(query)
                logger.info(f"Executed query: {query[:50]}...")
                return result
            except Exception as e:
                logger.error(f"Error executing query: {e}")
                raise

    async def _execute_with_plan(
        self, conn, query: str, capture_plan: bool = True
    ) -> Tuple[float, Optional[Dict], int]:
        if capture_plan and query.strip().upper().startswith("SELECT"):
            explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"

            start_time = time.perf_counter()
            plan_result = await conn.fetch(explain_query)
            end_time = time.perf_counter()

            execution_time_ms = (end_time - start_time) * 1000
            plan_data = plan_result[0]["QUERY PLAN"][0]
            rows_returned = plan_data.get("Actual Rows", 0)

            return execution_time_ms, plan_data, rows_returned
        else:
            start_time = time.perf_counter()
            result = await conn.fetch(query)
            end_time = time.perf_counter()

            execution_time_ms = (end_time - start_time) * 1000
            rows_returned = len(result) if result else 0

            return execution_time_ms, None, rows_returned

    async def benchmark_query(
        self,
        query: str,
        description: str,
        num_runs: int = 5,
        clear_cache_between_runs: bool = True,
        capture_plan: bool = True,
        warm_up_runs: int = 1,
        disable_seqscan: bool = False,
    ) -> QueryBenchmarkResult:
        result = QueryBenchmarkResult(
            query=query,
            query_hash="",
            description=description,
            execution_results=[],
        )

        async with self.pool.acquire() as conn:
            if disable_seqscan:
                await conn.execute("SET enable_seqscan TO off;")
                logger.info("Sequential scans disabled for this benchmark")

            if warm_up_runs > 0:
                logger.info(f"Running {warm_up_runs} warm-up runs for: {description}")
                for i in range(warm_up_runs):
                    try:
                        await self._execute_with_plan(conn, query, capture_plan=False)
                    except Exception as e:
                        logger.warning(f"Warm-up run {i + 1} failed: {e}")

            if clear_cache_between_runs:
                await self.clear_cache()
                await asyncio.sleep(0.5)

            logger.info(f"Running {num_runs} measurement runs for: {description}")
            for run in range(num_runs):
                if clear_cache_between_runs and run > 0:
                    await self.clear_cache()
                    await asyncio.sleep(0.5)
                    if disable_seqscan:
                        await conn.execute("SET enable_seqscan TO off;")

                try:
                    exec_time, plan, rows = await self._execute_with_plan(
                        conn, query, capture_plan
                    )

                    result.execution_results.append(
                        QueryExecutionResult(
                            execution_time_ms=exec_time,
                            query_plan=plan,
                            rows_returned=rows,
                            execution_number=run + 1,
                            cached=run > 0 and not clear_cache_between_runs,
                        )
                    )
                    logger.info(f"Run {run + 1}: {exec_time:.2f}ms")

                except Exception as e:
                    logger.error(f"Error in run {run + 1}: {e}")

            if disable_seqscan:
                await conn.execute("SET enable_seqscan TO on;")
                logger.info("Sequential scan setting restored to default")

        result.calculate_metrics()
        self.momento.store_result(result)
        return result

    def get_momento(self) -> QueryMomento:
        return self.momento

    def compare_last_two_results(self) -> Optional[Dict[str, Any]]:
        results = self.momento.get_all_results()
        if len(results) < 2:
            return None

        result1 = results[-2]
        result2 = results[-1]

        return self.momento.compare_results(
            result1.query_hash,
            result1.description,
            result2.query_hash,
            result2.description,
        )


class BenchmarkReporter:
    @staticmethod
    def generate_single_query_report(result: QueryBenchmarkResult) -> str:
        report = []
        report.append(f"Query: {result.description}")
        report.append(f"Hash: {result.query_hash}")
        report.append(f"SQL: {result.query}")
        report.append(f"Runs: {result.total_runs}")
        report.append(f"Average: {result.avg_time_ms:.2f}ms")
        report.append(f"Median: {result.median_time_ms:.2f}ms")
        report.append(f"Min: {result.min_time_ms:.2f}ms")
        report.append(f"Max: {result.max_time_ms:.2f}ms")
        report.append(f"Std Dev: {result.std_dev_ms:.2f}ms")
        report.append(f"Scan Type: {result.primary_scan_type}")
        report.append(f"Estimated Cost: {result.estimated_cost}")
        return "\n".join(report)

    @staticmethod
    def generate_comparison_report(comparison: Dict[str, Any]) -> str:
        if "error" in comparison:
            return f"Error: {comparison['error']}"

        report = []
        report.append("PERFORMANCE COMPARISON")
        report.append("=" * 50)
        report.append(f"Baseline: {comparison['baseline']['description']}")
        report.append(f"  - Average: {comparison['baseline']['avg_time_ms']:.2f}ms")
        report.append(f"  - Scan: {comparison['baseline']['scan_type']}")
        report.append("")
        report.append(f"Optimized: {comparison['optimized']['description']}")
        report.append(f"  - Average: {comparison['optimized']['avg_time_ms']:.2f}ms")
        report.append(f"  - Scan: {comparison['optimized']['scan_type']}")
        report.append("")
        report.append(f"Improvement: {comparison['improvement_percent']:.1f}%")
        report.append(f"Winner: {comparison['faster']}")
        return "\n".join(report)

    @staticmethod
    def generate_markdown_report(results: List[QueryBenchmarkResult]) -> str:
        report = []
        report.append("# PostgreSQL Query Benchmark Report")
        report.append("")
        report.append(f"**Generated:** {datetime.now().isoformat()}")
        report.append(f"**Generated by:** {os.getenv('USER', 'unknown')}")
        report.append(f"**Total Queries:** {len(results)}")
        report.append("")

        report.append("## Summary")
        report.append("")
        report.append(
            "| Query | Avg Time (ms) | Median (ms) | Std Dev (ms) | Scan Type |"
        )
        report.append(
            "|-------|---------------|-------------|--------------|-----------|"
        )

        for result in results:
            report.append(
                f"| {result.description} | {result.avg_time_ms:.2f} | {result.median_time_ms:.2f} | {result.std_dev_ms:.2f} | {result.primary_scan_type} |"
            )

        report.append("")
        report.append("## Detailed Results")
        report.append("")

        for i, result in enumerate(results, 1):
            report.append(f"### {i}. {result.description}")
            report.append("")
            report.append(f"**Query Hash:** `{result.query_hash}`")
            report.append("")
            report.append(f"**SQL Query:**")
            report.append(f"```sql")
            report.append(f"{result.query}")
            report.append(f"```")
            report.append("")
            report.append(f"**Performance Metrics:**")
            report.append(f"- Average: {result.avg_time_ms:.2f}ms")
            report.append(f"- Median: {result.median_time_ms:.2f}ms")
            report.append(f"- Min: {result.min_time_ms:.2f}ms")
            report.append(f"- Max: {result.max_time_ms:.2f}ms")
            report.append(f"- Standard Deviation: {result.std_dev_ms:.2f}ms")
            report.append(f"- Total Runs: {result.total_runs}")
            report.append("")
            report.append(f"**Query Plan:**")
            report.append(f"- Primary Scan Type: {result.primary_scan_type}")
            report.append(f"- Estimated Cost: {result.estimated_cost}")
            report.append(f"- Actual Cost: {result.actual_cost}")
            report.append("")

        return "\n".join(report)


async def quick_benchmark(
    query: str,
    description: str,
    num_runs: int = 5,
    connection_string: Optional[str] = None,
    disable_seqscan: bool = False,
) -> QueryBenchmarkResult:
    async with PostgreSQLBenchmark(connection_string) as benchmark:
        return await benchmark.benchmark_query(
            query, description, num_runs=num_runs, disable_seqscan=disable_seqscan
        )


async def execute_ddl(ddl_query: str, connection_string: Optional[str] = None) -> Any:
    async with PostgreSQLBenchmark(connection_string) as benchmark:
        return await benchmark.execute_query_only(ddl_query)


async def example_usage():
    async with PostgreSQLBenchmark() as benchmark:
        result1 = await benchmark.benchmark_query(
            query="SELECT * FROM users WHERE email = 'test@example.com'",
            description="User lookup by email - NO INDEX",
            num_runs=5,
        )

        await benchmark.execute_query_only(
            "CREATE INDEX idx_users_email ON users(email);"
        )

        result2 = await benchmark.benchmark_query(
            query="SELECT * FROM users WHERE email = 'test@example.com'",
            description="User lookup by email - WITH INDEX",
            num_runs=5,
        )

        comparison = benchmark.compare_last_two_results()
        if comparison:
            print(BenchmarkReporter.generate_comparison_report(comparison))

        all_results = benchmark.get_momento().get_all_results()
        markdown_report = BenchmarkReporter.generate_markdown_report(all_results)

        with open("benchmark_results.md", "w") as f:
            f.write(markdown_report)

        return all_results


if __name__ == "__main__":
    asyncio.run(example_usage())
