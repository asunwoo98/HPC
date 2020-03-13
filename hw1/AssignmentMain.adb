
with MatrixMult;
with text_io;

procedure AssignmentMain is
   use text_io;
   package int_io is new integer_io(integer);
   use int_io;
   use MatrixMult;

   task Reader1;

   task Reader2 is 
      entry start;
   end Reader2;

   task Printer is
      entry start;
   end Printer;

   A,B,C: sq_mat;
   input_complete: boolean:=False;

task body Reader1 is
      next_int: integer;
begin
   for x in 1..SIZE loop
      for y in 1..SIZE loop
         get(next_int);
         A(x,y):= next_int;
      end loop;
   end loop;
   Reader2.start;
end Reader1;

task body Reader2 is
      next_int: integer;
begin
   accept start;
   for x in 1..SIZE loop
      for y in 1..SIZE loop
         get(next_int);
         B(x,y):= next_int;
      end loop;
   end loop;
   input_complete:=True;
end Reader2;

task body Printer is
begin
   accept start;
--   for x in 1..SIZE loop
--      for y in 1..SIZE loop
--         put(A(x,y));
--      end loop;
--      new_line;
--   end loop;
--
--   new_line;
--
--  for x in 1..SIZE loop
--      for y in 1..SIZE loop
--         put(B(x,y));
--      end loop;
--      new_line;
--   end loop;
--
--   new_line;

   for x in 1..SIZE loop
      for y in 1..SIZE loop
         put(C(x,y));
      end loop;
      new_line;
   end loop;

end Printer;

begin
   while not input_complete
   loop
      null;
   end loop;
   MatMult(A,B,C);
   Printer.start;
end AssignmentMain;

