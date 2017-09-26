----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 19.08.2017 21:02:54
-- Design Name: 
-- Module Name: sweep_timer - Behavioral
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity sweep_timer is
    Port ( clk : in STD_LOGIC;
           rst : in STD_LOGIC;
           adf_done : in STD_LOGIC;
           sample_valid : out STD_LOGIC;
           write_timer : in STD_LOGIC;
           write_data : in STD_LOGIC_VECTOR (31 downto 0);
           write_delay : in STD_LOGIC;
           write_decimate : in STD_LOGIC;
           write_pa_off : in STD_LOGIC;
           pa_off : out STD_LOGIC);
end sweep_timer;

architecture Behavioral of sweep_timer is

-- (N-1)/2, where N = number of taps on the first FIR filter
constant FIR1_DELAY : integer := 60;
constant ADC_DELAY : integer := 6;
constant VALID_DELAY : integer := FIR1_DELAY+ADC_DELAY;

signal sweep_count : unsigned(15 downto 0) := to_unsigned(0, 16);
signal decimate_sweeps : unsigned(15 downto 0) := to_unsigned(0, 16);

signal count : unsigned(31 downto 0) := to_unsigned(0, 32);
signal sweep_top : unsigned(31 downto 0) := to_unsigned(40000, 32);
signal delay_top : unsigned(31 downto 0) := to_unsigned(20000, 32);
signal pa_off_top : unsigned(31 downto 0) := to_unsigned(15000, 32);

signal count_delay : std_logic := '0';

signal valid_pipe : std_logic_vector(VALID_DELAY-1 downto 0) := (others=> '0');
signal adf_done_prev : std_logic := '0';

signal decimate_block : std_logic;

signal pa_off_delay : std_logic := '1';

begin

timer_process : process(clk, rst)

begin

if rst = '1' then
    count <= to_unsigned(0, 32);
    
elsif rising_edge(clk) then

    if write_timer = '1' then
        sweep_top <= unsigned(write_data);
    elsif write_delay = '1' then
        delay_top <= unsigned(write_data);
    elsif write_decimate = '1' then
        decimate_sweeps <= unsigned(write_data(15 downto 0));
        sweep_count <= (others => '0');
    elsif write_pa_off = '1' then
        pa_off_top <= unsigned(write_data);
    end if;
    
    count <= count + 1;
    adf_done_prev <= adf_done;
    
    if count_delay = '0' then
        if count = sweep_top then
            count <= to_unsigned(0, 32);
            count_delay <= not count_delay;
            if pa_off_top /= to_unsigned(0, 32) then
                pa_off_delay <= '1';
            end if;
            if sweep_count /= decimate_sweeps then
                sweep_count <= sweep_count + 1;
            else
                sweep_count <= (others => '0');
            end if;
        end if;
    else
        if count = delay_top then
            count <= to_unsigned(0, 32);
            count_delay <= not count_delay;
            pa_off_delay <= '0';
        end if;
        if count = pa_off_top then
            pa_off_delay <= '0';
        end if;
    end if;

    -- When sweep is done start counting delay
    if adf_done = '1' and adf_done_prev = '0' then
        count <= to_unsigned(0, 32);
        count_delay <= '1'; -- Delay
        pa_off_delay <= '1';
        
        if sweep_count /= decimate_sweeps then
            sweep_count <= sweep_count + 1;
        else
            sweep_count <= (others => '0');
        end if;
                    
    end if;
    
end if;

end process;

decimate_block <= '0' when sweep_count = to_unsigned(0, 16) else '1';

pa_off <= pa_off_delay or decimate_block;

sample_valid <= valid_pipe(VALID_DELAY - 1);


delay_process : process(clk)
begin

if rising_edge(clk) then

    valid_pipe(0) <= (not count_delay) and (not decimate_block);
    
    for i in 1 to VALID_DELAY - 1 loop
        valid_pipe(i) <= valid_pipe(i-1);
    end loop;
    
end if;

end process;


end Behavioral;
