with text_io;
use text_io;

procedure A is
    package int_io is new integer_io(integer);
    use int_io;

    task loop1 is
        entry start;
    end loop1;
    task loop2 is
        entry start;
    end loop2;

    task body loop1 is
    begin
        for x in 1..100 loop
            if x mod 10=1 then
                accept start;
            end if;
            put(x);
            if x mod 10=0 then
                loop2.start;
            end if;
        end loop;
    end loop1;
    task body loop2 is
    begin
        for x in 201..300 loop
            if x mod 10=1 then
                accept start;
            end if;
            put(x);
            if x mod 10=0 then
                loop1.start;
            end if;
        end loop;
    end loop2;
begin
	loop1.start;
end A;