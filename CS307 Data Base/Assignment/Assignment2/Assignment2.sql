-- Question1
--/////////////////////////////////////////////////////
SELECT s.english_name, s.latitude FROM line_detail ld
INNER JOIN stations s on s.station_id = ld.station_id
WHERE ld.line_id = 4;

SELECT concat('The latitude of '|| temp.english_name || ' station is:')
FROM(SELECT s.english_name, s.latitude FROM line_detail ld
INNER JOIN stations s on s.station_id = ld.station_id
WHERE ld.line_id = 4) temp;

SELECT CASE
            WHEN temp.latitude IS NULL
                THEN concat('No latitude information for '|| temp.english_name || ' station')
            ELSE
                concat('The latitude of '|| temp.english_name || ' station is: ' || temp.latitude )
       END AS latitude_info
FROM(SELECT s.english_name, s.latitude FROM line_detail ld
        INNER JOIN stations s on s.station_id = ld.station_id
        WHERE ld.line_id = 4) temp;

-- Question2
--/////////////////////////////////////////////////////

SELECT bl.station_id
FROM bus_lines bl
WHERE bl.bus_line = '2';


SELECT ld.line_id,s.chinese_name
FROM line_detail ld
INNER JOIN stations s on s.station_id = ld.station_id
WHERE ld.station_id IN(
    SELECT bl.station_id
    FROM bus_lines bl
    WHERE bl.bus_line = '2')
ORDER BY line_id, ld.station_id;


-- Question3
--/////////////////////////////////////////////////////
SELECT DISTINCT line_id
FROM line_detail;

SELECT MAX(temp.count)
FROM(
    SELECT line_id, count(station_id) AS count
    FROM line_detail
    GROUP BY line_id
    ) temp;



-- 找到拥有最多 station 的 line id
SELECT temp.line_id
FROM (
            SELECT line_id, count(station_id) AS count
            FROM line_detail
            GROUP BY line_id
) temp
WHERE temp.count = (SELECT MAX(temp1.count)
                    FROM(SELECT line_id, count(station_id) AS count
                    FROM line_detail
                    GROUP BY line_id) temp1)
ORDER BY temp.line_id;



SELECT ld.line_id, s.chinese_name
FROM line_detail ld
INNER JOIN stations s on s.station_id = ld.station_id
WHERE ld.line_id IN(
                SELECT temp.line_id
                FROM (
                        SELECT line_id, count(station_id) AS count
                        FROM line_detail
                        GROUP BY line_id
                ) temp
                WHERE temp.count = (SELECT MAX(temp1.count)
                    FROM(SELECT line_id, count(station_id) AS count
                    FROM line_detail
                    GROUP BY line_id) temp1)
    ) AND ld.num = 1
ORDER BY ld.line_id;




-- Question4
--/////////////////////////////////////////////////////

SELECT line_id, opening
FROM lines
WHERE opening BETWEEN 2008 AND 2021;

-- 每个线的开通时间 2008-2021
SELECT temp.opening AS year, concat('line '|| temp.line_id || ' opened') AS comment
FROM(SELECT line_id, opening
    FROM lines
    WHERE opening BETWEEN 2008 AND 2021) temp
ORDER BY year;

-- 每条线的拓展时间
SELECT temp.latest_extension AS year, concat('line '|| temp.line_id || ' extended') AS comment
FROM (
         SELECT line_id, latest_extension
         FROM lines
         WHERE latest_extension BETWEEN 2008 AND 2021
     ) temp
ORDER BY year;


SELECT temp.opening AS year, concat('line '|| temp.line_id || ' opened') AS comment
FROM(SELECT line_id, opening
    FROM lines
    WHERE opening BETWEEN 2008 AND 2021) temp


UNION

SELECT temp.latest_extension AS year, concat('line '|| temp.line_id || ' extended') AS comment
FROM (
         SELECT line_id, latest_extension
         FROM lines
         WHERE latest_extension BETWEEN 2008 AND 2021
     ) temp
ORDER BY year;




-- Question5
--/////////////////////////////////////////////////////

-- 二号线经过的站点(去重)
SELECT DISTINCT station_id
FROM line_detail ld
WHERE ld.line_id = 2
ORDER BY station_id;


SELECT DISTINCT station_id
FROM line_detail ld
WHERE ld.line_id = 5
ORDER BY station_id;


SELECT DISTINCT ld.station_id, s.chinese_name
FROM line_detail ld
INNER JOIN stations s on s.station_id = ld.station_id
WHERE ld.station_id IN(
    SELECT DISTINCT station_id
    FROM line_detail ld
    WHERE ld.line_id = 5
    ORDER BY station_id
    )
    AND ld.station_id IN(
    SELECT DISTINCT station_id
    FROM line_detail ld
    WHERE ld.line_id = 1
    ORDER BY station_id
    )
    AND ld.station_id NOT IN(
        SELECT DISTINCT station_id
        FROM line_detail ld
        WHERE ld.line_id = 2
    )
ORDER BY ld.station_id;