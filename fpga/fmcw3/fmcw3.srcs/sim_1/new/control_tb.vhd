----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 14.08.2017 21:07:17
-- Design Name: 
-- Module Name: control_tb - Behavioral
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

entity control_tb is
--  Port ( );
end control_tb;

architecture Behavioral of control_tb is

constant clk_period : time := 25 ns;
signal clk, rst : std_logic;
signal data_in : std_logic_vector(7 downto 0);
signal data_valid, ready, led : std_logic;


signal data_count : integer := 0;

constant PACKET_LENGTH : integer := 5;
type memory_type is array (0 to PACKET_LENGTH-1) of std_logic_vector(7 downto 0);
signal memory : memory_type := (
0 => "10101010",
1 => "00000001",
2 => "00000000",
others => (others => '0'));

begin

control : entity work.control
    Port map ( clk => clk,
           rst => rst,
           data_in => data_in,
           ready => ready,
           data_valid => data_valid,
           led => led);

rst <= '0';

-- Clock process definitions
clk_process : process
begin
    clk <= '0';
    wait for clk_period/2;
    clk <= '1';
    wait for clk_period/2;
end process;

comm_process : process(clk, ready)
begin

if rising_edge(clk) then
    data_in <= memory(data_count);
    if ready = '1' then
        data_valid <= '1';
        if data_count = PACKET_LENGTH-1 then
            data_count <= 0;
        else
            data_count <= data_count + 1;
        end if;
    end if;
end if;

end process;


end Behavioral;
