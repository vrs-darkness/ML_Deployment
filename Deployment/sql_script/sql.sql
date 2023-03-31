create database EMP ;

use EMP;

create table Employee (EmpID int, EmpName varchar(30),Passwd varchar(30) );


insert into Employee(EMPID,EmpName,Passwd) values(1,'Vignesh','test123');

insert into Employee(EMPID,EmpName,Passwd) values(2,'Raam','test321');

select * from Employee ;