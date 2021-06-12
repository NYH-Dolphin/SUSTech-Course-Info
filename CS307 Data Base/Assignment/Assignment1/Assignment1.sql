-- 请把schema设置成SZmetro



-- question1
-- -----------------------------------------------------------
SELECT DISTINCT s.station_id, s.chinese_name
FROM stations s
WHERE s.chinese_name LIKE '%山%'
ORDER BY s.station_id;
-- -----------------------------------------------------------


-- question2
-- -----------------------------------------------------------
SELECT ci.name,ci.hex

FROM(
    SELECT ROW_NUMBER () OVER (ORDER BY hex ASC) AS ids, c.name, c.hex
    FROM color_names c
    ORDER BY c.hex) as ci

WHERE ci.ids BETWEEN 5 AND 9;


-- 为color_name表加一个id
SELECT ROW_NUMBER () OVER (ORDER BY hex ASC) AS ids, c.name, c.hex
FROM color_names c;
-- -----------------------------------------------------------


-- question3
-- -----------------------------------------------------------
SELECT b.bus_line, count(b.station_id) AS count
FROM bus_lines b
GROUP BY b.bus_line;
-- -----------------------------------------------------------


-- question4
-- -----------------------------------------------------------
SELECT MAX(bs.count)
FROM(
    SELECT b.bus_line, count(b.station_id) AS count
    FROM bus_lines b
    GROUP BY b.bus_line) AS bs;
-- -----------------------------------------------------------

-- question5
-- -----------------------------------------------------------
SELECT bs.bus_line, bs.count AS cnt
FROM (
         SELECT b.bus_line, count(b.station_id) AS count
         FROM bus_lines b
         GROUP BY b.bus_line
         ORDER BY count DESC
     ) AS bs
WHERE bs.count = (
    SELECT MAX(bb.count)
    FROM(
        SELECT count(b.station_id) AS count
        FROM bus_lines b
        GROUP BY b.bus_line) AS bb
    );

-- -----------------------------------------------------------

-- question6
-- -----------------------------------------------------------
SELECT DISTINCT b.station_id
FROM bus_lines b
WHERE b.bus_line = '1' OR b.bus_line = '2'
ORDER BY b.station_id;
-- -----------------------------------------------------------


-- question7
-- -----------------------------------------------------------
SELECT DISTINCT b.station_id
FROM bus_lines b
WHERE b.bus_line = '1'
AND b.station_id IN (SELECT DISTINCT b.station_id
                        FROM bus_lines b
                        WHERE b.bus_line = '2'
                        ORDER BY b.station_id)
ORDER BY b.station_id;
-- -----------------------------------------------------------


-- question8
-- -----------------------------------------------------------

SELECT

CAST(


    (SELECT (
        CAST((SELECT COUNT(one_two)
                FROM (
                      SELECT DISTINCT b.station_id
                      FROM bus_lines b
                      WHERE b.bus_line = '1'
                      AND b.station_id IN (SELECT DISTINCT b.station_id
                                             FROM bus_lines b
                                             WHERE b.bus_line = '2'
                                             ORDER BY b.station_id)
                      ORDER BY b.station_id
                ) AS one_two) AS DECIMAL(10,2))
            /

        (SELECT COUNT(one)
        FROM (
            SELECT DISTINCT b.station_id
            FROM bus_lines b
            WHERE b.bus_line = '1'
            ORDER BY b.station_id) AS one)

           ))


    AS DECIMAL(10,2) ) AS round;

SELECT COUNT(one)
FROM(
SELECT DISTINCT b.station_id
FROM bus_lines b
WHERE b.bus_line = '1'
ORDER BY b.station_id) AS one;



SELECT COUNT(one_two)
FROM(
    SELECT DISTINCT b.station_id
    FROM bus_lines b
    WHERE b.bus_line = '1'
    AND b.station_id IN (SELECT DISTINCT b.station_id
                        FROM bus_lines b
                        WHERE b.bus_line = '2'
                        ORDER BY b.station_id)
ORDER BY b.station_id

        ) AS one_two


-- -----------------------------------------------------------


