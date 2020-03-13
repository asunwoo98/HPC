package MatrixMult is
   SIZE: integer:=10;
   type sq_mat is array (1..SIZE, 1..SIZE) of integer;
   procedure MatMult(A:in sq_mat; B:in sq_mat; C:out sq_mat);
end MatrixMult;