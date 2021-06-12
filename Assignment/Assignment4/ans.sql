CREATE OR REPLACE FUNCTION password_check()
    RETURNS TRIGGER
AS
$$
DECLARE
    uppercase       INTEGER = 0;
    lowercase       INTEGER = 0;
    digits          INTEGER = 0;
    passwordLen     INTEGER;
    pointer         INTEGER = 1;
    asciiCode       INTEGER;
    updateTime      INTEGER;
    firstID         INTEGER = 0;
    lastPassword1   VARCHAR;
    lastPassword2   VARCHAR;
    lastPassword3   VARCHAR;
    boo1            BOOLEAN;
    boo2            BOOLEAN;
    boo3            BOOLEAN;


BEGIN

    IF (NEW.id <= 9999999 OR NEW.id > 100000000) THEN
        RAISE EXCEPTION 'The length of student ID should be 8';
    END IF;


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


    IF (LENGTH(NEW.password) < 8) THEN
        RAISE EXCEPTION 'The length of the password is less than 8 digits';
    END IF;


    IF (POSITION(NEW.username IN NEW.password) <> 0) THEN
        RAISE EXCEPTION 'The password cannot contain user''s username';
    END IF;

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


    firstID = NEW.id / 10000000;
    IF (firstID = 1) THEN
        NEW.role = 'School students';
    ELSEIF (firstID = 2) THEN
        NEW.role = 'Exchange students';
    ELSEIF (firstID = 3 OR firstID = 5) THEN
        NEW.role = 'Teacher';
    END IF;


    SELECT COUNT(*)
    FROM account_log
    WHERE user_id = NEW.id
    INTO updateTime;

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

    IF (boo1 = TRUE OR boo2 = TRUE OR boo3 = TRUE) THEN
        RAISE EXCEPTION 'fail, the new password is the same as the previous two passwords';
    END IF;


    NEW.password = crypt(NEW.password, gen_salt('bf'));
    RETURN NEW;
END
$$ LANGUAGE plpgsql;


CREATE OR REPLACE FUNCTION on_update()
    RETURNS TRIGGER
AS
$$
BEGIN
    INSERT INTO account_log SELECT NEW.id, CURRENT_TIMESTAMP, OLD.password;
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

CREATE TRIGGER password_trigger
    BEFORE INSERT OR UPDATE
    ON account
    FOR EACH ROW
EXECUTE PROCEDURE password_check();

CREATE TRIGGER updatePwd_trigger
    AFTER UPDATE
    ON account
    FOR EACH ROW
EXECUTE PROCEDURE on_update();