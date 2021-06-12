# 1. 用户优先级管理

## 用户种类表

|              | 创建数据库 | 角色修改 | 流复制 | SELECT | CREATE | DELETE | ALTER |
| ------------ | ---------- | -------- | ------ | ------ | ------ | ------ | ----- |
| SUPERUSER    | ✔          | ✔        | ✔      | ✔      | ✔      | ✔      | ✔     |
| CREATEDB     | ✔          |          |        |        |        |        |       |
| CREATROLE    |            | ✔        |        |        |        |        |       |
| REPLICATION  |            |          | ✔      |        |        |        |       |
| GRANT SELECT |            |          |        | ✔      |        |        |       |
| GRANT CREATE |            |          |        |        | ✔      |        |       |
| GRANT DELETE |            |          |        |        |        | ✔      |       |
| GRANT ALTER  |            |          |        |        |        |        | ✔     |
|              |            |          |        |        |        |        |       |

### 超级用户

```mysql
CREATE USER super_user WITH PASSWORD '20010922nyh' SUPERUSER;
```

### 可创建数据库用户

```mysql
CREATE USER createDB_user WITH PASSWORD '20010922nyh' CREATEDB;
```

### 可以删除用户的用户

```mysql
CREATE USER createRole_user WITH PASSWORD '20010922nyh' CREATEROLE;
```

### 可选高级用户

- SUPERUSER
- NOSUPERUSER
- CREATEDB
- NOCREATEDB
- CREATEROLE
- NOCREATEROLE
- INHERIT
- NOINHERIT
- LOGIN
- NOLOGIN
- REPLICATION
- NOREPLICATION
- BYPASSRLS
- NOBYPASSRLS
- CONNECTION LIMIT connlimit

### 普通用户

```mysql
CREATE USER user1 WITH PASSWORD '20010922nyh' LOGIN; 
GRANT ALL PRIVILEGES ON DATABASE postgres TO user1;
GRANT ALL PRIVILEGES ON TABLE project1.class TO user1;
# 授权SELECT
GRANT SELECT ON project1.class TO user2;
# 取消授权SELECT
REVOKE SELECT ON project1.class FROM user2;
```

## 使用场景

|                         | 创建数据库 | 角色修改 | 流复制 | SELECT | CREATE                                                       | DELETE                                                       | ALTER                                                        |
| ----------------------- | ---------- | -------- | ------ | ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 创建者+管理者（最高层） | ✔          | ✔        | ✔      | ✔ all  | ✔ all                                                        | ✔ all                                                        | ✔ all                                                        |
| 数据库管理员            |            |          | ✔      | ✔ all  | ✔ all                                                        | ✔ all                                                        | ✔ all                                                        |
| 教务系统管理员          |            |          |        | ✔ all  | ✔<br />course<br />class<br />sport<br />distribution<br />teacher-class<br />student-course<br />week<br />prerequisite | ✔<br />course<br />class<br />sport<br />distribution<br />teacher-class<br />student-course<br />week<br />prerequisite | ✔<br />course<br />class<br />sport<br />distribution<br />teacher-class<br />student-course<br />week<br />prerequisite |
| 老师                    |            |          |        | ✔ all  | ✔<br />class<br />course<br />sport<br />teacher-class<br />distribution<br />prerequisite |                                                              | ✔<br />class<br />course<br />sport<br />teacher-distribution<br />prerequisite |
| 辅导员                  |            |          |        | ✔ all  |                                                              |                                                              |                                                              |
| 学生                    |            |          |        | ✔ all  | ✔<br />student-course                                        | ✔<br />student-course                                        | ✔<br />student-course                                        |

# 2.索引

按照下列标准建立索引的列

- 频繁搜索的列
- 经常用作查询选择的列
- 经常排序、分组的列
- 经常用作连接的列

不要使用下面的列创建索引

- 仅包含几个不同值的列
- 表中仅包含几行

除去所有已经设置好的主键索引和唯一索引外，对于本project应该设置索引的地方有

class表

- 联合索引
    course_id, language, teaching_object, capacity, class_name, comment

course表

- 联合索引
    course_id, name, hour, cridit, department_id

teacher_class表

- 联合索引
    teacher_name, class_id, course_id, class_name

# 3. 用户需求设计

## 辅导员

### 查看本书院学生名单 1s

```mysql
SELECT * FROM student
WHERE college_id = <书院id>;
```

### 查看本书院学生的选课情况 10-12s

```mysql
SELECT s.school_number, c.course_id 
FROM student s
INNER JOIN student_course sc on s.school_number = sc.student_id
INNER JOIN course c on c.id = sc.course_id
WHERE college_id = <书院id>;
```

### 查看本书院选课少于指定学分的学生 6-8s

```mysql
SELECT s.school_number FROM student s
INNER JOIN student_course sc on s.school_number = sc.student_id
INNER JOIN course c on c.id = sc.course_id
WHERE college_id = <书院id>
GROUP BY s.school_number
HAVING sum(credit) < <指定学分>;
```

## 学生

### 查看自己选课的相关信息 100ms

```mysql
SELECT c.course_id, c.name, c.hour, c.credit, d.name AS department FROM student s
INNER JOIN student_course sc on s.school_number = sc.student_id
INNER JOIN course c on c.id = sc.course_id
INNER JOIN department d on c.department_id = d.id
WHERE s.school_number = <学生id>;
```

### 查看自己的已选学分 100ms

```mysql
SELECT sum(c.credit) AS credit_sum FROM student s
INNER JOIN student_course sc on s.school_number = sc.student_id
INNER JOIN course c on c.id = sc.course_id
WHERE s.school_number = <学生id>
GROUP BY s.school_number;
```

### 查看全校课表 100ms

```mysql
SELECT course.course_id AS course_ID, course.name AS name,
       course.hour AS hour, course.credit AS credit,
       c.class_name AS class_name, c.language AS language, t.name,
       c.teaching_object AS teaching_object, c.property AS property,
       c.capacity AS capacity, d.class_time AS time,
       d.week_day AS week_day, l.comment AS location
FROM course
INNER JOIN class c on course.course_id = c.course_id
INNER JOIN distribution d on c.id = d.class_id
INNER JOIN location l on d.location_id = l.id
INNER JOIN teacher_class tc on c.id = tc.class_id
INNER JOIN teacher t ON tc.teacher_id = t.id;
```

### 查看指定院系课表 100ms

```mysql
SELECT course.course_id AS course_ID, course.name AS name,
       course.hour AS hour, course.credit AS credit,
       c.class_name AS class_name, c.language AS language, t.name,
       c.teaching_object AS teaching_object, c.property AS property,
       c.capacity AS capacity, d.class_time AS time,
       d.week_day AS week_day, l.comment AS location
FROM course
INNER JOIN class c on course.course_id = c.course_id
INNER JOIN distribution d on c.id = d.class_id
INNER JOIN location l on d.location_id = l.id
INNER JOIN teacher_class tc on c.id = tc.class_id
INNER JOIN teacher t ON tc.teacher_id = t.id
WHERE course.department_id = <10>;
```

### 查看指定时间安排的课程 100ms

```mysql
SELECT course.course_id AS course_ID, course.name AS name,
       course.hour AS hour, course.credit AS credit,
       c.class_name AS class_name, c.language AS language, t.name,
       c.teaching_object AS teaching_object, c.property AS property,
       c.capacity AS capacity, d.class_time AS time,
       d.week_day AS week_day, l.comment AS location
FROM course
INNER JOIN class c on course.course_id = c.course_id
INNER JOIN distribution d on c.id = d.class_id
INNER JOIN location l on d.location_id = l.id
INNER JOIN teacher_class tc on c.id = tc.class_id
INNER JOIN teacher t ON tc.teacher_id = t.id
WHERE week_day = <3> AND class_time = <'5-6'>;
```

### 查看指定课程的详细信息 100ms

```mysql
SELECT course.course_id AS course_ID, course.name AS name,
       course.hour AS hour, course.credit AS credit,
       c.class_name AS class_name, c.language AS language, t.name,
       c.teaching_object AS teaching_object, c.property AS property,
       c.capacity AS capacity, d.class_time AS time,
       d.week_day AS week_day, l.comment AS location
FROM course
INNER JOIN class c on course.course_id = c.course_id
INNER JOIN distribution d on c.id = d.class_id
INNER JOIN location l on d.location_id = l.id
INNER JOIN teacher_class tc on c.id = tc.class_id
INNER JOIN teacher t ON tc.teacher_id = t.id
WHERE c.course_id = <'CS306'>;
```

### 查看自己的已选课程 100ms

```mysql
SELECT course.course_id AS course_ID, course.name AS name,
       course.hour AS hour, course.credit AS credit,
       c.class_name AS class_name, c.language AS language, t.name,
       c.teaching_object AS teaching_object, c.property AS property,
       c.capacity AS capacity, d.class_time AS time,
       d.week_day AS week_day, l.comment AS location
FROM course
INNER JOIN class c on course.course_id = c.course_id
INNER JOIN distribution d on c.id = d.class_id
INNER JOIN location l on d.location_id = l.id
INNER JOIN teacher_class tc on c.id = tc.class_id
INNER JOIN teacher t ON tc.teacher_id = t.id
INNER JOIN student_course sc on course.id = sc.course_id
WHERE sc.student_id = 11000004;
```

### 查找需要选的课是否已经存在 100ms

```mysql
SELECT <'CS401'>
IN( SELECT c.course_id
    FROM student_course
    INNER JOIN course c on c.id = student_course.course_id
    WHERE student_id = 11000004);
```

## 老师

### 查看自己的授课安排 100ms

```mysql
SELECT tc.course_id AS course_id,c2.name,c2.credit,c2.hour, c.class_name, c.class_number,
       c.language, c.teaching_object, c.property,c.capacity,
       d.week_day,d.class_time,l.comment
FROM teacher_class tc
INNER JOIN class c on c.id = tc.class_id
INNER JOIN distribution d on c.id = d.class_id
INNER JOIN location l on d.location_id = l.id
INNER JOIN week w on d.week_id = w.id
INNER JOIN course c2 on c.course_id = c2.course_id
WHERE teacher_id = <29>;
```

### 查看选择这门课的学生 1-3s

```mysql
SELECT student_id
FROM student_course
WHERE course_id = <13>
```

```mysql
SELECT s.school_number, s.name
FROM student_course
INNER JOIN student s on s.school_number = student_course.student_id
WHERE course_id = <13>
```