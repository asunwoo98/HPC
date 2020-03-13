
with text_io;
use text_io;

package body MatrixMult is

procedure MatMult(A:in sq_mat; B:in sq_mat; C:out sq_mat) is

   task type mat_mult_task is
      entry SetCell(i_index:integer; j_index:integer);
      entry GetValue(Val: out Integer);
   end mat_mult_task;
   
   task body mat_mult_task is
      i,j:integer;
      sum:integer:=0;
   begin
      accept SetCell(i_index:integer; j_index:integer) do
         i:=i_index;
         j:=j_index;
      end SetCell;
      
      for index in 1..SIZE loop
         sum:= sum + A(i,index)*B(index,j);
      end loop;
      
      accept GetValue(Val: out Integer) do
         Val:= sum;
      end GetValue;

   end mat_mult_task;

   mat_tasks: array(1..SIZE, 1..SIZE) of mat_mult_task;

begin -- MatMult
   for x in 1..SIZE loop
      for y in 1..SIZE loop
         mat_tasks(x,y).SetCell(x,y);
      end loop;
   end loop;

   for x in 1..SIZE loop
      for y in 1..SIZE loop
         mat_tasks(x,y).GetValue(C(x,y));
      end loop;
   end loop;

end MatMult;

end MatrixMult;