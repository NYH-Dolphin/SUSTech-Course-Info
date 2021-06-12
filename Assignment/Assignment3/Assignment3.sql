-- Question1
-- 请求出在三号线上不在四号线上，且维度在平均之上的车站有多少个
-- ////////////////////////////////////////////////////////////



SELECT station_id FROM line_detail
WHERE line_id = 4;


SELECT sum(latitude)/count(latitude)
FROM stations;

SELECT station_id FROM stations
WHERE latitude < (
    SELECT sum(latitude)/count(latitude)
    FROM stations
    );


SELECT count(*) FROM (
                         SELECT station_id l
                         FROM line_detail ld
                         WHERE line_id = 3
                           AND ld.station_id NOT IN (
                             SELECT station_id
                             FROM line_detail
                             WHERE line_id = 4
                         )
                           AND ld.station_id NOT IN (
                             SELECT station_id
                             FROM stations
                             WHERE latitude < (
                                 SELECT sum(latitude) / count(latitude)
                                 FROM stations
                             )
                         )
                     ) temp;


-- Question2
-- 请找出所有站点的英文名首字母中出现频率最高的字母，然后按区给出含有这个字母的地铁站数量的排名
-- ////////////////////////////////////////////////////////////

-- 找到最大的出现频率
                  SELECT temp.str, count(*) cnt
                  FROM (
                           SELECT substring(english_name, 1, 1) str
                           FROM stations
                       ) temp
                  GROUP BY temp.str
                  ORDER BY count(*) DESC;


-- 找到最大的那个字母
SELECT *
FROM(
    SELECT temp.str, count(*) cnt
                  FROM (
                           SELECT substring(english_name, 1, 1) str
                           FROM stations
                       ) temp
                  GROUP BY temp.str
                  ORDER BY count(*) DESC
        ) temp1
WHERE temp1.cnt = (
    SELECT max(temp1.cnt) FROM (
                  SELECT temp.str, count(*) cnt
                  FROM (
                           SELECT substring(english_name, 1, 1) str
                           FROM stations
                       ) temp
                  GROUP BY temp.str
                  ORDER BY count(*) DESC
              ) temp1
    );


-- S
SELECT temp1.str
FROM(
    SELECT temp.str, count(*) cnt
                  FROM (
                           SELECT substring(english_name, 1, 1) str
                           FROM stations
                       ) temp
                  GROUP BY temp.str
                  ORDER BY count(*) DESC
        ) temp1
WHERE temp1.cnt = (
    SELECT max(temp1.cnt) FROM (
                  SELECT temp.str, count(*) cnt
                  FROM (
                           SELECT substring(english_name, 1, 1) str
                           FROM stations
                       ) temp
                  GROUP BY temp.str
                  ORDER BY count(*) DESC
              ) temp1
    );

-- 首字母包含S， 错误答案
SELECT district, count(*)
FROM stations
WHERE substring(english_name, 1, 1) IN(
    SELECT temp1.str
    FROM(
        SELECT temp.str, count(*) cnt
            FROM (
                SELECT substring(english_name, 1, 1) str
                FROM stations
                 ) temp
            GROUP BY temp.str
            ORDER BY count(*) DESC
        ) temp1
    WHERE temp1.cnt = (
        SELECT max(temp1.cnt) FROM (
            SELECT temp.str, count(*) cnt
            FROM (
                SELECT substring(english_name, 1, 1) str
                FROM stations
                 ) temp
            GROUP BY temp.str
            ORDER BY count(*) DESC
            ) temp1
        )
    )
GROUP BY district
ORDER BY count(*) DESC;



-- 近似答案1
SELECT *
FROM(
SELECT district, count(*)
FROM stations
WHERE english_name like '%s%'
OR english_name like '%S%'
GROUP BY district
ORDER BY count(*) DESC, district
    ) temp2
WHERE length(temp2.district) <> 0;



-- 答案
SELECT temp.district, count(*)
FROM (
                  SELECT *
                  FROM (
                           SELECT regexp_split_to_array(english_name, '') @> array [(SELECT temp1.str
FROM(
    SELECT temp.str, count(*) cnt
                  FROM (
                           SELECT substring(english_name, 1, 1) str
                           FROM stations
                       ) temp
                  GROUP BY temp.str
                  ORDER BY count(*) DESC
        ) temp1
WHERE temp1.cnt = (
    SELECT max(temp1.cnt) FROM (
                  SELECT temp.str, count(*) cnt
                  FROM (
                           SELECT substring(english_name, 1, 1) str
                           FROM stations
                       ) temp
                  GROUP BY temp.str
                  ORDER BY count(*) DESC
              ) temp1
    ))] as result,
                                  english_name,
                                  district                                                  district
                           FROM stations
                       ) temp33
                  WHERE temp33.result = true
                  UNION

                  SELECT *
                  FROM (
                           SELECT regexp_split_to_array(english_name, '') @> array [(SELECT lower(temp1.str)
FROM(
    SELECT temp.str, count(*) cnt
                  FROM (
                           SELECT substring(english_name, 1, 1) str
                           FROM stations
                       ) temp
                  GROUP BY temp.str
                  ORDER BY count(*) DESC
        ) temp1
WHERE temp1.cnt = (
    SELECT max(temp1.cnt) FROM (
                  SELECT temp.str, count(*) cnt
                  FROM (
                           SELECT substring(english_name, 1, 1) str
                           FROM stations
                       ) temp
                  GROUP BY temp.str
                  ORDER BY count(*) DESC
              ) temp1
    ))] as result,
                                  english_name,
                                  district                                                  district
                           FROM stations
                       ) temp33
                  WHERE temp33.result = true
              ) temp
WHERE length(temp.district) <> 0
GROUP BY temp.district
ORDER BY count(*) DESC, temp.district;


-- Question3
-- 请找出龙华，宝安，南山三个区内，拥有公交接驳站最多的 3 个地铁炸
-- 按照district排序 Longhua>Baoan>Nanshan，公交车站数量以及station_id
-- ////////////////////////////////////////////////////////////
SELECT s.district, s.station_id ,count(*)
FROM stations s
INNER JOIN bus_lines bl on s.station_id = bl.station_id
WHERE district IN('Longhua','Bao''an','Nanshan')
GROUP BY s.station_id
ORDER BY count(*) DESC;







(SELECT s.district ,count(*), s.station_id
FROM stations s
INNER JOIN bus_lines bl on s.station_id = bl.station_id
WHERE district = 'Longhua'
GROUP BY s.station_id
ORDER BY count(*) DESC, s.station_id DESC
LIMIT 3)

UNION ALL

(SELECT s.district ,count(*), s.station_id
FROM stations s
INNER JOIN bus_lines bl on s.station_id = bl.station_id
WHERE district = 'Bao''an'
GROUP BY s.station_id
ORDER BY count(*) DESC, s.station_id DESC
LIMIT 3)

UNION ALL

(SELECT s.district ,count(*), s.station_id
FROM stations s
INNER JOIN bus_lines bl on s.station_id = bl.station_id
WHERE district = 'Nanshan'
GROUP BY s.station_id
ORDER BY count(*) DESC, s.station_id DESC
LIMIT 3);


-- Question4
-- 每一条地铁线所经过的地铁站数目是一定的
-- 请输出到目前为止的所有地铁站id，开通时间，所经过的地铁站数
-- 以及每一条地铁线相较于上一条建的地铁线的地铁站数目变化率
-- 结果保留两位小数
-- ////////////////////////////////////////////////////////////


SELECT ld.line_id, count(*) cnt
FROM line_detail ld
GROUP BY ld.line_id;



SELECT temp.*, lines.opening
FROM(
    SELECT ld.line_id, count(*) cnt
    FROM line_detail ld
    GROUP BY ld.line_id
    ) temp
INNER JOIN lines ON temp.line_id = lines.line_id
ORDER BY lines.opening, temp.line_id;


SELECT temp.*, lines.opening, lead(temp.cnt,-1) OVER w as nextCategory
FROM(
    SELECT ld.line_id, count(*) cnt
    FROM line_detail ld
    GROUP BY ld.line_id
    ) temp
INNER JOIN lines ON temp.line_id = lines.line_id
WINDOW w AS (
        ORDER BY lines.opening, temp.line_id
    )
ORDER BY lines.opening, temp.line_id;


SELECT temp2.line_id, temp2.opening, temp2.cnt, concat(round(1-(temp2.cnt::numeric/temp2.next),2)*100,'%') FROM (
                  SELECT temp.*, lines.opening, lead(temp.cnt, -1) OVER w as next
                  FROM (
                           SELECT ld.line_id, count(*) cnt
                           FROM line_detail ld
                           GROUP BY ld.line_id
                       ) temp
                           INNER JOIN lines ON temp.line_id = lines.line_id
                      WINDOW w AS (
                          ORDER BY lines.opening, temp.line_id
                          )
                  ORDER BY lines.opening, temp.line_id
              ) temp2;







(SELECT temp2.line_id, temp2.opening, temp2.cnt, round(1-(temp2.cnt::numeric/temp2.next),4)rnd FROM (
                  SELECT temp.*, lines.opening, lead(temp.cnt, -1) OVER w as next
                  FROM (
                           SELECT ld.line_id, count(*) cnt
                           FROM line_detail ld
                           GROUP BY ld.line_id
                       ) temp
                           INNER JOIN lines ON temp.line_id = lines.line_id
                      WINDOW w AS (
                          ORDER BY lines.opening, temp.line_id
                          )
                  ORDER BY lines.opening, temp.line_id
              ) temp2
LIMIT 1);




(SELECT temp.line_id, lines.opening, temp.cnt,cast(lead(temp.cnt,-1) OVER w AS VARCHAR )AS next
FROM(
    SELECT ld.line_id, count(*) cnt
    FROM line_detail ld
    GROUP BY ld.line_id
    ) temp
INNER JOIN lines ON temp.line_id = lines.line_id
WINDOW w AS (
        ORDER BY lines.opening, temp.line_id
    )
ORDER BY lines.opening, temp.line_id
LIMIT 1)

UNION ALL


(SELECT temp3.line_id, temp3.opening, temp3.cnt, concat(round(temp3.rnd * 100,2), '%') FROM (
                  SELECT temp2.line_id, temp2.opening, temp2.cnt, round((temp2.cnt::numeric / temp2.next) - 1 , 4) rnd
                  FROM (
                           SELECT temp.*, lines.opening, lead(temp.cnt, -1) OVER w as next
                           FROM (
                                    SELECT ld.line_id, count(*) cnt
                                    FROM line_detail ld
                                    GROUP BY ld.line_id
                                ) temp
                                    INNER JOIN lines ON temp.line_id = lines.line_id
                               WINDOW w AS (
                                   ORDER BY lines.opening, temp.line_id
                                   )
                           ORDER BY lines.opening, temp.line_id
                       ) temp2
              ) temp3
WHERE temp3.rnd IS NOT NULL)
