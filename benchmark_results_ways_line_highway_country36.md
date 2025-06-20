# PostgreSQL Query Benchmark Report

**Generated:** 2025-06-20T04:33:11.070740
**Generated by:** krschap
**Total Queries:** 2

## Summary

| Query | Avg Time (ms) | Median (ms) | Std Dev (ms) | Scan Type |
|-------|---------------|-------------|--------------|-----------|
| Geometry lookup by gist index - country 36 | 1002.81 | 999.47 | 20.36 |  |
| Geometry lookup by h3 index - country 36 | 228.73 | 228.92 | 2.69 |  |

## Detailed Results

### 1. Geometry lookup by gist index - country 36

**Query Hash:** `926dbe03`

**SQL Query:**
```sql
WITH country_geom AS (
            SELECT geometry as geom
            FROM countries
            WHERE cid = 36
            ),
            filtered_by_geom AS (
            SELECT w.*
            FROM ways_line w
            JOIN country_geom c ON ST_Intersects(w.geom, c.geom)
            )
            SELECT count(*)
            FROM filtered_by_geom
            WHERE tags ? 'highway';
```

**Performance Metrics:**
- Average: 1002.81ms
- Median: 999.47ms
- Min: 978.09ms
- Max: 1031.80ms
- Standard Deviation: 20.36ms
- Total Runs: 5

**Query Plan:**
- Primary Scan Type: 
- Estimated Cost: 0
- Actual Cost: 0

### 2. Geometry lookup by h3 index - country 36

**Query Hash:** `2f0f7ce5`

**SQL Query:**
```sql
WITH filtered_by_h3 AS (
            SELECT t.*
            FROM ways_line t
            JOIN country_h3_flat ch ON t.h3 = ch.h3_index
            WHERE ch.country_id = 36
            )
            SELECT count(*)
            FROM filtered_by_h3
            WHERE tags ? 'highway';
            
```

**Performance Metrics:**
- Average: 228.73ms
- Median: 228.92ms
- Min: 225.45ms
- Max: 232.76ms
- Standard Deviation: 2.69ms
- Total Runs: 5

**Query Plan:**
- Primary Scan Type: 
- Estimated Cost: 0
- Actual Cost: 0
