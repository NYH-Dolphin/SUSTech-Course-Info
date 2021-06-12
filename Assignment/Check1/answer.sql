-- Question1
-- ///////////////////////////////////////////////////////

-- window function
SELECT c.country_name,m.title,m.year_released,
            max(m.year_released) OVER(PARTITION BY c.country_name) AS most_recent
    FROM movies m
    INNER JOIN countries c on m.country = c.country_code
    WHERE c.continent = 'ASIA';

SELECT temp.country_name, temp.title, temp.year_released AS year
FROM(
    SELECT c.country_name,m.title,m.year_released,
            max(m.year_released) OVER(PARTITION BY c.country_name) AS most_recent
    FROM movies m
    INNER JOIN countries c on m.country = c.country_code
    WHERE c.continent = 'ASIA') temp
WHERE temp.year_released = temp.most_recent;


-- Question2
-- ///////////////////////////////////////////////////////

-- country 的名字和它们的电影的计数
SELECT c.country_name, count(m.movieid) AS cnt
FROM movies m
INNER JOIN countries c on c.country_code = m.country
GROUP BY m.country, c.country_name;

-- 计算平均数
SELECT avg(cnt)
FROM(
    SELECT c.country_name, count(m.movieid) AS cnt
    FROM movies m
    INNER JOIN countries c on c.country_code = m.country
    GROUP BY m.country, c.country_name) temp;

-- 通过平均数进行筛选
SELECT * FROM(
    SELECT c.country_name, count(m.movieid) AS cnt
    FROM movies m
    INNER JOIN countries c on c.country_code = m.country
    GROUP BY m.country, c.country_name) table1
WHERE table1.cnt > (
    SELECT avg(cnt)
    FROM(
        SELECT c.country_name, count(m.movieid) AS cnt
        FROM movies m
        INNER JOIN countries c on c.country_code = m.country
        GROUP BY m.country, c.country_name) temp
    )
ORDER BY cnt DESC;


-- Question3
-- ///////////////////////////////////////////////////////

-- 记录每个电影的排名
SELECT m.title, m.year_released,
       rank() OVER(ORDER BY year_released DESC) rnk
FROM movies m
WHERE m.country = 'cn'
ORDER BY year_released DESC;


SELECT temp.title, temp.year_released AS year
FROM(
    SELECT m.title, m.year_released,
       rank() OVER(ORDER BY year_released DESC) rnk
    FROM movies m
    WHERE m.country = 'cn'
    ORDER BY year_released DESC
        )temp
WHERE temp.rnk <= 10;

