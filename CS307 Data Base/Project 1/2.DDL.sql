create table project1.course_info
(
    totalcapacity integer,
    courseid      varchar(400),
    prerequisite  varchar(400),
    coursehour    integer,
    coursecredit  integer,
    coursename    varchar(400),
    classname     varchar(400),
    coursedept    varchar(400),
    teacher       varchar(400),
    weeklist1     varchar(400),
    location1     varchar(400),
    classtime1    varchar(400),
    weekday1      integer,
    weeklist2     varchar(400),
    location2     varchar(400),
    classtime2    varchar(400),
    weekday2      integer,
    weeklist3     varchar(400),
    location3     varchar(400),
    classtime3    varchar(400),
    weekday3      integer
);

alter table project1.course_info
    owner to postgres;

create table project1.select_course
(
    name    varchar(400),
    gender  varchar(400),
    college varchar(400),
    number  integer,
    c1      varchar(400),
    c2      varchar(400),
    c3      varchar(400),
    c4      varchar(400),
    c5      varchar(400),
    c6      varchar(400)
);

alter table project1.select_course
    owner to postgres;

create table project1.teacher
(
    id   serial       not null
        constraint teacher_pkey
            primary key,
    name varchar(100) not null
        constraint teacher_name_key
            unique
);

alter table project1.teacher
    owner to postgres;

create table project1.department
(
    id   serial       not null
        constraint department_pkey
            primary key,
    name varchar(100) not null
        constraint department_name_key
            unique
);

alter table project1.department
    owner to postgres;

create table project1.course
(
    id            integer      not null
        constraint course_pkey
            primary key,
    course_id     varchar(100) not null
        constraint course_course_id_key
            unique,
    name          varchar(100) not null,
    hour          integer,
    credit        integer,
    department_id integer
        constraint course_department_id_fk
            references project1.department
);

alter table project1.course
    owner to postgres;

create index course_id_index
    on project1.course (course_id);

create table project1.college
(
    id           integer      not null
        constraint college_pkey
            primary key,
    chinese_name varchar(100) not null,
    english_name varchar(100) not null,
    constraint college_chinese_name_english_name_key
        unique (chinese_name, english_name)
);

alter table project1.college
    owner to postgres;

create table project1.student
(
    school_number integer      not null
        constraint student_pkey
            primary key,
    name          varchar(100) not null,
    gender        char,
    college_id    integer
        constraint student_college_id_fk
            references project1.college
);

alter table project1.student
    owner to postgres;

create index nameindex
    on project1.student (name);

create index college_id_and_school_number_and_name_index
    on project1.student (college_id, school_number, name);

create table project1.sport
(
    id           integer not null
        constraint sport_pkey
            primary key,
    class_number integer,
    course_id    varchar not null
        constraint sport_course_course_id_fk
            references project1.course (course_id),
    class_type   varchar(100),
    capacity     integer,
    comment      varchar(400),
    constraint sport_class_type_class_number_key
        unique (class_type, class_number)
);

alter table project1.sport
    owner to postgres;

create table project1.location
(
    id       integer not null
        constraint location_pkey
            primary key,
    area     varchar(100),
    building varchar(100),
    room     varchar(100),
    function varchar(100),
    comment  varchar(400),
    constraint location_area_building_room_key
        unique (area, building, room)
);

alter table project1.location
    owner to postgres;

create table project1.class
(
    id              integer not null
        constraint class_pkey
            primary key,
    class_number    integer,
    course_id       varchar not null
        constraint class_course_course_id_fk
            references project1.course (course_id),
    language        varchar(100),
    teaching_object varchar(100),
    property        varchar(100),
    makeup          varchar(100),
    capacity        integer,
    class_name      varchar(100),
    comment         varchar(400),
    constraint class_class_number_course_id_language_key
        unique (class_number, course_id, language)
);

alter table project1.class
    owner to postgres;

grant select on project1.class to user2;

create table project1.teacher_class
(
    teacher_id integer not null
        constraint teacher_class_teacher_id_fk
            references project1.teacher,
    class_id   integer not null
        constraint teacher_class_class_id_fk
            references project1.class,
    constraint teacher_class_pkey
        primary key (teacher_id, class_id)
);

alter table project1.teacher_class
    owner to postgres;

create table project1.week
(
    id      integer not null
        constraint week_pkey
            primary key,
    w1      integer,
    w2      integer,
    w3      integer,
    w4      integer,
    w5      integer,
    w6      integer,
    w7      integer,
    w8      integer,
    w9      integer,
    w10     integer,
    w11     integer,
    w12     integer,
    w13     integer,
    w14     integer,
    w15     integer,
    w16     integer,
    comment varchar(400),
    constraint week_w1_w2_w3_w4_w5_w6_w7_w8_w9_w10_w11_w12_w13_w14_w15_w16_key
        unique (w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16)
);

alter table project1.week
    owner to postgres;

create table project1.student_course
(
    student_id integer not null
        constraint student_course_student_school_number_fk
            references project1.student,
    course_id  integer not null
        constraint student_course_course_id_fk
            references project1.course,
    constraint student_curse_pkey
        primary key (student_id, course_id)
);

alter table project1.student_course
    owner to postgres;

create table project1.distribution
(
    id          integer not null
        constraint distribution_pk
            primary key,
    class_id    integer
        constraint distribution_class_id_fk
            references project1.class,
    week_id     integer
        constraint distribution_week_id_fk
            references project1.week,
    location_id integer
        constraint distribution_location_id_fk
            references project1.location,
    class_time  varchar(100),
    week_day    integer,
    sport_id    integer
        constraint distribution_sport_id_fk
            references project1.sport,
    constraint distribution_class_id_week_id_location_id_class_time_week_d_key
        unique (class_id, week_id, location_id, class_time, week_day)
);

alter table project1.distribution
    owner to postgres;

create table project1.prerequisite
(
    id          integer not null
        constraint prerequisite_or_pkey
            primary key,
    course_id_b integer
        constraint prerequisite_class_id_fk_2
            references project1.course,
    group_id    integer,
    course_id_a integer
        constraint prerequisite_course_id_fk
            references project1.course,
    constraint prerequisite_or_class_id_a_class_id_b_group_id_key
        unique (course_id_a, course_id_b, group_id)
);

alter table project1.prerequisite
    owner to postgres;

