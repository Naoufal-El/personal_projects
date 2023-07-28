import psycopg2

from conn_db import connect

class EmployeeGateway:
    @staticmethod
    def add(id_emp, ename, job, mgr, hiredate, sal, comm, deptno):
        conn = None
        try:
            conn = connect()
            cur = conn.cursor()
            cur.execute(
                """
                SELECT 1 FROM dept WHERE deptno = %s
                """,
                (id_emp,)
            )
            res = cur.fetchone()
            if res is None:
                cur.execute(
                    """
                    INSERT INTO dept (deptno, dname, loc)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (id_emp, ename, job, mgr, hiredate, sal, comm, deptno)
                )
                conn.commit()
                print("Employee added successfully.")
            else:
                print("Employee already exists.")
            cur.close()

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

        finally:
            if conn is not None:
                conn.close()

    @staticmethod
    def update(id_emp, new_emp=None, new_job=None, new_mgr=None, new_hiredate=None, new_sal=None, new_comm=None, new_deptno=None):
        conn = None
        updates = []
        try:
            conn = connect()
            cur = conn.cursor()
            if new_emp:
                updates.append(f"emp = {new_emp}")
            if new_job:
                updates.append(f"job = {new_job}")
            if new_mgr:
                updates.append(f"mgr = {new_mgr}")
            if new_hiredate:
                updates.append(f"hiredate = {new_hiredate}")
            if new_sal:
                updates.append(f"sal = {new_sal}")
            if new_comm:
                updates.append(f"comm = {new_comm}")
            if new_deptno:
                updates.append(f"deptno = {new_deptno}")

            update_query = f"UPDATE emp SET {','.join(updates)} WHERE id = {id_emp}"
            cur.execute(update_query)
            conn.commit()
            cur.close()

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

        finally:
            if conn is not None:
                conn.close()

    @staticmethod
    def delete(id_emp):
        conn = None
        try:
            conn = connect()
            cur = conn.cursor()
            cur.execute("DELETE FROM emp WHERE id = %s", (id_emp,))
            conn.commit()
            cur.close()

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

        finally:
            if conn is not None:
                conn.close()

    @staticmethod
    def search(name):
        conn = None
        try:
            conn = connect()
            cur = conn.cursor()
            cur.execute("SELECT * FROM emp WHERE ename = %s", (name,))
            row = cur.fetchone()

            if row:
                print(row)
            else:
                print("The name", name, "does not exist")
            cur.close()

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

        finally:
            if conn is not None:
                conn.close()