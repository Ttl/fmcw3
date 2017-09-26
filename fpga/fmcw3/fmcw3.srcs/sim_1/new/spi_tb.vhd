----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 29.07.2017 14:51:44
-- Design Name: 
-- Module Name: spi_tb - Behavioral
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

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
--use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity spi_tb is
--  Port ( );
end spi_tb;

architecture Behavioral of spi_tb is

signal clk, rst : std_logic;
signal write, busy, spi_clk, spi_data, spi_le, ack : std_logic := '0';
signal data_in : std_logic_vector(31 downto 0) := (others => '0');

constant clk_period : time := 10 ns;

begin

spi : entity work.spi
    Generic map(SPI_CLK_DIVIDER => 1)
Port map ( clk => clk,
       rst => rst,
       data_in => data_in,
       write => write,
       busy => busy,
       spi_clk => spi_clk,
       spi_data => spi_data,
       spi_le => spi_le,
       ack => ack);

test_process : process
begin
    data_in <= "10001010101010101010101010111110";
    write <= '0';
    wait for 10*clk_period;
    write <= '1';
    wait until ack = '1';
    write <= '0';
    wait for clk_period;
end process;

rst_process : process
begin
    rst <= '1';
    wait for clk_period;
    rst <= '0';
    wait;
end process;

-- Clock process definitions
clk_process : process
begin
    clk <= '0';
    wait for clk_period/2;
    clk <= '1';
    wait for clk_period/2;
end process;

end Behavioral;
