from Department import DepartmentGateway
from Employee import EmployeeGateway


emp = EmployeeGateway()
dep = DepartmentGateway()


while True:
    print("Management App")
    print("==============")
    print("0)add department")
    print("1)delete department")
    print("2)add employee")
    print("3)update employee")
    print("4)delete employee")
    print("5)search employee")
    print("6)Exit")
    choice = input("Select your choice to proceed: ")

    if choice == "0":
        deptno = input("Insert dept Number: ")
        dname = input("Insert department name: ")
        loc = input("Insert location: ")
        dep.add(deptno, dname, loc)


    elif choice == "1":
        deptno = input("Insert the department Number to Proceed: ")
        dep.delete(deptno)
        print("Department deleted successfully")

    elif choice == "2":
        id = input("Enter id of Employee: ")
        ename = input("Enter name of Employee: ")
        job = input("Enter job: ")
        mgr = input("Enter mgr: ")
        hiredate = input("Enter hiredate: ")
        sal = input("Enter sal: ")
        comm = input("Enter comm: ")
        deptno = input("Enter deptno: ")
        emp.add(id, ename, job, mgr, hiredate, sal, comm, deptno)
        print("Employee added successfully")

    elif choice == "3":
        id = input("Enter Employee id to update: ")
        ename = input("Enter Employee name: ")
        job = input("Enter job: ")
        mgr = input("Enter mgr: ")
        hiredate = input("Enter a hire date: ")
        sal = input("Enter sal: ")
        comm = input("Enter comm: ")
        deptno = input("Enter deptno: ")
        emp.update(id, ename, job, mgr, hiredate, sal, comm, deptno)

        print("Employee updated successfully")

    elif choice == "4":
        id = input("Enter Employee id to delete: ")
        emp.delete(id)

        print("Employee deleted successfully")

    elif choice == "5":
        ename = input("Enter Employee name to search: ")
        emp.search(ename)

    elif choice == "6":
        print("Have a nice day !")
        break

    else:
        print("Invalid choice! Try again ")
