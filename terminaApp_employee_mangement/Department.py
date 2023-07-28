import psycopg2

from conn_db import connect


class DepartmentGateway:
    @staticmethod
    def add(deptno, dname, loc):
        conn = None
        try:
            conn = connect()
            cur = conn.cursor()
            cur.execute(
                """
                SELECT 1 FROM dept WHERE deptno = %s
                """,
                (deptno,)
            )
            res = cur.fetchone()
            if res is None:
                cur.execute(
                    """
                    INSERT INTO dept (deptno, dname, loc)
                    VALUES (%s, %s, %s)
                    """,
                    (deptno, dname, loc)
                )
                conn.commit()
                print("Department added successfully.")
            else:
                print("Department already exists.")
            cur.close()

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()

    @staticmethod
    def delete(deptno):
        conn = None
        try:
            conn = connect()
            cur = conn.cursor()
            conn.cursor()
            cur.execute("DELETE FROM dept WHERE deptno = %s", (deptno,))
            conn.commit()
            cur.close()

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

        finally:  # Close Connection
            if conn is not None:
                conn.close()
