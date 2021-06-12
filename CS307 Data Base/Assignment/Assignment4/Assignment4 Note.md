account

<img src="C:\Users\Lotus0922\AppData\Roaming\Typora\typora-user-images\image-20210430104953807.png" alt="image-20210430104953807" style="zoom:50%;" />

account_log

<img src="C:\Users\Lotus0922\AppData\Roaming\Typora\typora-user-images\image-20210430104953807.png" alt="image-20210430104953807" style="zoom:50%;" />



### 初始化

安装 prcrypto

```mysql
CREATE EXTENSION IF NOT EXISTS pgcrypto;
```

```mysql
CREATE TABLE IF NOT EXISTS account1
(
    id       INT PRIMARY KEY,
    username VARCHAR(20) NOT NULL,
    password VARCHAR(60) NOT NULL,
    role     VARCHAR(20)
);
CREATE TABLE IF NOT EXISTS account_log
(
    user_id     INT,
    update_date TIMESTAMP,
    password    VARCHAR(60) NOT NULL,
    PRIMARY KEY (user_id, update_date),
    FOREIGN KEY (user_id) REFERENCES account (id)
);
```

### 触发器

#### Password_check

```mysql
CREATE OR REPLACE FUNCTION password_check()
    RETURNS TRIGGER
AS
$$
DECLARE
    -- Q1 第一问 flag
    uppercase   INTEGER = 0;
    lowercase   INTEGER = 0;
    digits      INTEGER = 0;

    -- Q1 第四问 flag
    passwordLen INTEGER;
    pointer     INTEGER = 1;
    asciiCode   INTEGER;

    -- Q2
    firstID     INTEGER = 0; -- id的第一位

BEGIN
    -- ID checking
    -- The length of student ID should be 8,
    -- /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    IF (NEW.id <= 9999999 OR NEW.id > 100000000) THEN
        RAISE EXCEPTION 'The length of student ID should be 8';
    END IF;

    -- Password checking
    -- 1.The password contains both uppercase letters, lowercase letters, and digits.
    -- /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    -- 检查大写
    IF (POSITION('A' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('B' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('C' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('D' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('E' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('F' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('G' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('H' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('I' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('J' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('K' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('L' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('M' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('N' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('O' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('P' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('Q' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('R' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('S' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('T' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('U' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('V' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('W' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('X' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('Y' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('Z' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    -- 检查小写
    IF (POSITION('a' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('b' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('c' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('d' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('e' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('f' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('g' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('h' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('i' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('j' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('k' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('l' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('m' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('n' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('o' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('p' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('q' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('r' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('s' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('t' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('u' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('v' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('w' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('x' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('y' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('z' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    -- 检查数字
    IF (POSITION('0' IN NEW.password) <> 0) THEN digits = 1; END IF;
    IF (POSITION('1' IN NEW.password) <> 0) THEN digits = 1; END IF;
    IF (POSITION('2' IN NEW.password) <> 0) THEN digits = 1; END IF;
    IF (POSITION('3' IN NEW.password) <> 0) THEN digits = 1; END IF;
    IF (POSITION('4' IN NEW.password) <> 0) THEN digits = 1; END IF;
    IF (POSITION('5' IN NEW.password) <> 0) THEN digits = 1; END IF;
    IF (POSITION('6' IN NEW.password) <> 0) THEN digits = 1; END IF;
    IF (POSITION('7' IN NEW.password) <> 0) THEN digits = 1; END IF;
    IF (POSITION('8' IN NEW.password) <> 0) THEN digits = 1; END IF;
    IF (POSITION('9' IN NEW.password) <> 0) THEN digits = 1; END IF;

    IF (uppercase = 0 OR lowercase = 0 OR digits = 0) THEN
        RAISE EXCEPTION 'Password should contain both uppercase letters, lowercase letters, and digits';
    END IF;

    -- 2. The length of the password is larger than 8 digits.
    -- /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    IF (LENGTH(NEW.password) < 8) THEN
        RAISE EXCEPTION 'The length of the password is less than 8 digits';
    END IF;

    -- 3. The password cannot contain user's username
    -- /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    IF (POSITION(NEW.username IN NEW.password) <> 0) THEN
        RAISE EXCEPTION 'The password cannot contain user''s username';
    END IF;

    -- 4. Passwords cannot contain characters other than upper and lower case letters, digits,
    -- underscores( _ ), asterisks ( * ), and dollars ( $ ).
    -- /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    passwordLen = LENGTH(NEW.password);
    pointer = 1;
    WHILE (pointer <= passwordLen)
        LOOP
            asciiCode = ASCII(SUBSTR(NEW.password, pointer, 1));
            IF NOT ((asciiCode BETWEEN 48 AND 57) OR (asciiCode BETWEEN 65 AND 90) OR (asciiCode BETWEEN 97 AND 122)
                OR (asciiCode = 42) OR (asciiCode = 95) OR (asciiCode = 36)) THEN
                RAISE EXCEPTION 'Passwords cannot contain characters other than upper and lower case letters, digits,
underscores( _ ), asterisks ( * ), and dollars ( $ ) for character: %, ascii: %' , SUBSTR(NEW.password, pointer, 1), asciiCode;
            END IF;
            pointer = pointer + 1;

        END LOOP;

    --  Analyze the role of account
    -- We can make sure that the first digit of ID in test cases only can be 1,2,3 or 5.
    -- /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    firstID = NEW.id / 10000000;
    IF (firstID = 1) THEN
        NEW.role = 'School students';
    ELSEIF (firstID = 2) THEN
        NEW.role = 'Exchange students';
    ELSEIF (firstID = 3 OR firstID = 5) THEN
        NEW.role = 'Teacher';
    END IF;

    NEW.password = crypt(NEW.password, gen_salt('bf'));
    RETURN NEW;
END ;
$$ LANGUAGE plpgsql;
```

#### password_trigger

```mysql
CREATE TRIGGER password_trigger
    BEFORE INSERT
    ON account
    FOR EACH ROW
EXECUTE PROCEDURE password_check();
```

#### 测试样例

```mysql
INSERT INTO account(id, username, password)
VALUES (30020824, 'Molly', 'abAB12e');-- less than 8 digits


INSERT INTO account(id, username, password)
VALUES (50020825, 'Harmony', 'abcdefgh'); -- only lowercase


INSERT INTO account(id, username, password)
VALUES (30020826, 'Bright', 'Abcaac12'); -- success

INSERT INTO account(id, username, password)
VALUES (50020827, 'Firm', 'abc123asd'); -- no uppercase

INSERT INTO account(id, username, password)
VALUES (11913558, 'Roswell', '123578AbdeFa'); -- success

INSERT INTO account(id, username, password)
VALUES (22011249, 'Robust', 'asdaA12999999'); -- success

INSERT INTO account(id, username, password)
VALUES (11937541, 'Ross', 'Ba1sad19a'); -- success

INSERT INTO account(id, username, password)
VALUES (11991451, 'Sirena', 'ACzc901a'); -- success

INSERT INTO account(id, username, password)
VALUES (11841390, 'Frederick', 'asdasdzc12esLsadz'); -- success

INSERT INTO account(id, username, password)
VALUES (11751923, 'Wesley', 'aszcAsa12sd'); -- success

INSERT INTO account(id, username, password)
VALUES (50019827, 'Lloyd', 'zxcA21_9asd'); -- success

INSERT INTO account(id, username, password)
VALUES (50019981, 'Bob', '123Bob2ads'); -- include user's own name

INSERT INTO account(id, username, password)
VALUES (30891203, 'Smith', '_asd*91Ab'); -- success

INSERT INTO account(id, username, password)
VALUES (11913724, 'Zoo', 'Abc12&}asad'); -- include invalid character "}"

INSERT INTO account(id, username, password)
VALUES (1194567, 'HUANG', 'hasd19Hasd');
-- too short
```

#### On_update

```mysql
CREATE OR REPLACE FUNCTION on_update()
    RETURNS TRIGGER
AS
$$
DECLARE
    -- Q1 第一问 flag
    uppercase   INTEGER = 0;
    lowercase   INTEGER = 0;
    digits      INTEGER = 0;

    -- Q1 第四问 flag
    passwordLen INTEGER;
    pointer     INTEGER = 1;
    asciiCode   INTEGER;


    updateTime    INTEGER;
    -- 最近 3 次 密码
    lastPassword1 VARCHAR; -- 倒数第一次密码
    lastPassword2 VARCHAR; -- 倒数第二次密码
    lastPassword3 VARCHAR; -- 倒数第三次密码
    boo1          BOOLEAN;
    boo2          BOOLEAN;
    boo3          BOOLEAN;


BEGIN
        -- ID checking
    -- The length of student ID should be 8,
    -- /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    IF (NEW.id <= 9999999 OR NEW.id > 100000000) THEN
        RAISE EXCEPTION 'The length of student ID should be 8';
    END IF;

    -- Password checking
    -- 1.The password contains both uppercase letters, lowercase letters, and digits.
    -- /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    -- 检查大写
    IF (POSITION('A' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('B' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('C' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('D' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('E' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('F' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('G' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('H' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('I' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('J' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('K' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('L' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('M' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('N' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('O' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('P' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('Q' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('R' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('S' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('T' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('U' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('V' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('W' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('X' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('Y' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    IF (POSITION('Z' IN NEW.password) <> 0) THEN uppercase = 1; END IF;
    -- 检查小写
    IF (POSITION('a' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('b' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('c' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('d' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('e' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('f' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('g' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('h' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('i' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('j' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('k' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('l' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('m' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('n' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('o' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('p' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('q' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('r' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('s' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('t' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('u' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('v' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('w' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('x' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('y' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    IF (POSITION('z' IN NEW.password) <> 0) THEN lowercase = 1; END IF;
    -- 检查数字
    IF (POSITION('0' IN NEW.password) <> 0) THEN digits = 1; END IF;
    IF (POSITION('1' IN NEW.password) <> 0) THEN digits = 1; END IF;
    IF (POSITION('2' IN NEW.password) <> 0) THEN digits = 1; END IF;
    IF (POSITION('3' IN NEW.password) <> 0) THEN digits = 1; END IF;
    IF (POSITION('4' IN NEW.password) <> 0) THEN digits = 1; END IF;
    IF (POSITION('5' IN NEW.password) <> 0) THEN digits = 1; END IF;
    IF (POSITION('6' IN NEW.password) <> 0) THEN digits = 1; END IF;
    IF (POSITION('7' IN NEW.password) <> 0) THEN digits = 1; END IF;
    IF (POSITION('8' IN NEW.password) <> 0) THEN digits = 1; END IF;
    IF (POSITION('9' IN NEW.password) <> 0) THEN digits = 1; END IF;

    IF (uppercase = 0 OR lowercase = 0 OR digits = 0) THEN
        RAISE EXCEPTION 'Password should contain both uppercase letters, lowercase letters, and digits';
    END IF;

    -- 2. The length of the password is larger than 8 digits.
    -- /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    IF (LENGTH(NEW.password) < 8) THEN
        RAISE EXCEPTION 'The length of the password is less than 8 digits';
    END IF;

    -- 3. The password cannot contain user's username
    -- /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    IF (POSITION(NEW.username IN NEW.password) <> 0) THEN
        RAISE EXCEPTION 'The password cannot contain user''s username';
    END IF;

    -- 4. Passwords cannot contain characters other than upper and lower case letters, digits,
    -- underscores( _ ), asterisks ( * ), and dollars ( $ ).
    -- /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    passwordLen = LENGTH(NEW.password);
    pointer = 1;
    WHILE (pointer <= passwordLen)
        LOOP
            asciiCode = ASCII(SUBSTR(NEW.password, pointer, 1));
            IF NOT ((asciiCode BETWEEN 48 AND 57) OR (asciiCode BETWEEN 65 AND 90) OR (asciiCode BETWEEN 97 AND 122)
                OR (asciiCode = 42) OR (asciiCode = 95) OR (asciiCode = 36)) THEN
                RAISE EXCEPTION 'Passwords cannot contain characters other than upper and lower case letters, digits,
underscores( _ ), asterisks ( * ), and dollars ( $ ) for character: %, ascii: %' , SUBSTR(NEW.password, pointer, 1), asciiCode;
            END IF;
            pointer = pointer + 1;

        END LOOP;




    -- 第二问部分代码****************************************************************************************
    -- write your sql here
    SELECT COUNT(*)
    FROM account_log
    WHERE user_id = NEW.id
    INTO updateTime;

    -- 超过 3 次更新
    IF updateTime >= 3 THEN
        SELECT temp.password
        FROM (
                 SELECT ROW_NUMBER() OVER (ORDER BY update_date DESC) AS rank, password
                 FROM account_log
                 WHERE user_id = NEW.id
                 LIMIT 3) temp
        WHERE temp.rank = 1
        INTO lastPassword1;

        SELECT temp.password
        FROM (
                 SELECT ROW_NUMBER() OVER (ORDER BY update_date DESC) AS rank, password
                 FROM account_log
                 WHERE user_id = NEW.id
                 LIMIT 3) temp
        WHERE temp.rank = 2
        INTO lastPassword2;

        SELECT temp.password
        FROM (
                 SELECT ROW_NUMBER() OVER (ORDER BY update_date DESC) AS rank, password
                 FROM account_log
                 WHERE user_id = NEW.id
                 LIMIT 3) temp
        WHERE temp.rank = 3
        INTO lastPassword3;

        SELECT (lastPassword1 = crypt(NEW.password, lastPassword1)) INTO boo1;
        SELECT (lastPassword2 = crypt(NEW.password, lastPassword2)) INTO boo2;
        SELECT (lastPassword3 = crypt(NEW.password, lastPassword3)) INTO boo3;

        --RAISE EXCEPTION 'boo2: %, boo3: %', boo2, boo3;
        IF (boo2 = TRUE OR boo3 = TRUE) THEN
            RAISE EXCEPTION 'fail, the new password is the same as the previous two passwords';
        END IF;

    ELSEIF updateTime = 2 THEN
        SELECT temp.password
        FROM (
                 SELECT ROW_NUMBER() OVER (ORDER BY update_date DESC) AS rank, password
                 FROM account_log
                 WHERE user_id = NEW.id
                 LIMIT 3) temp
        WHERE temp.rank = 1
        INTO lastPassword1;

        SELECT temp.password
        FROM (
                 SELECT ROW_NUMBER() OVER (ORDER BY update_date DESC) AS rank, password
                 FROM account_log
                 WHERE user_id = NEW.id
                 LIMIT 3) temp
        WHERE temp.rank = 2
        INTO lastPassword2;

        SELECT (lastPassword1 = crypt(NEW.password, lastPassword1)) INTO boo1;
        SELECT (lastPassword2 = crypt(NEW.password, lastPassword2)) INTO boo2;

        IF (boo2 = TRUE) THEN
            RAISE EXCEPTION 'fail, the new password is the same as the previous passwords';
        END IF;


    END IF;


    INSERT INTO account_log SELECT NEW.id, CURRENT_TIMESTAMP, crypt(NEW.password, gen_salt('bf'));
    RETURN NEW;
END
$$ LANGUAGE plpgsql;
```

#### updatePwd_trigger

```mysql
CREATE TRIGGER updatePwd_trigger
    BEFORE UPDATE
    ON account
    FOR EACH ROW
-- unaware of the new table structure
EXECUTE PROCEDURE on_update();
```

#### 测试样例

```mysql
-- 初始化
INSERT INTO account(id, username, password)
VALUES (30020826, 'HAHAHA', 'Abcaac12');
-- 初始化 UPDATE
UPDATE account
SET password = 'Abcaac12'
WHERE id = 30020826;


-- Initial Password (id=30020826): Abcaac12
UPDATE account
SET password = ''
WHERE id = 30020826; -- fail, the new password is too short

UPDATE account
SET password = 'Absazxc1213'
WHERE id = 30020826; -- success, the new password is different from the old "Absazxc1213"

UPDATE account
SET password = 'Abcaac12'
WHERE id = 30020826; -- fail, the new password is the same as one of the previous three passwords

UPDATE account
SET password = 'Absazxc1213'
WHERE id = 30020826; -- success, the new password is the same as last password, but log table has added a new row

UPDATE account
SET password = 'Absazxc1213'
WHERE id = 30020826; -- fail, the new password is the same as the previous two passwords

UPDATE account
SET password = 'Abasd12_a'
WHERE id = 30020826; -- success, the new is different from the previous three

UPDATE account
SET password = 'asd1sadA8z'
WHERE id = 30020826; -- success

UPDATE account
SET password = 'Absazxc1213'
WHERE id = 30020826; -- fail

UPDATE account
SET password = 'Abcaac12'
WHERE id = 30020826; -- success, the new is different from the previous three

UPDATE account
SET password = 'onjkn98_Q'
WHERE id = 30020826; -- success

UPDATE account
SET password = 'Absazxc1213'
WHERE id = 30020826;
-- success
-- Finally, password is equals to "Absazxc1213"
```