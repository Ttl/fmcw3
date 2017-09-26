----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 24.07.2017 15:09:18
-- Design Name: 
-- Module Name: adc_sim - Behavioral
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
use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity adc_sim is
end adc_sim;

architecture Behavioral of adc_sim is

signal data_a : std_logic_vector(13 downto 0) := (others => '0');
signal data_b : std_logic_vector(13 downto 0) := (others => '0');
signal data : std_logic_vector(11 downto 0) := (others => '0');
signal adc_valid : std_logic;

signal clk, rst : std_logic := '0';

-- Clock period definitions
constant clk_period : time := 10 ns;


begin

adc : entity work.adc
    Port map ( clk => clk,
           adc_data => data,
           rst => rst,
           data_a => data_a,
           data_b => data_b,
           valid => adc_valid);
           
data_process : process(clk)
begin
    if rising_edge(clk) or falling_edge(clk) then
        --data <= std_logic_vector(unsigned(data) + 1);
        data <= (11=> '0', others => '1');
    end if;
end process;

rst_process :process
begin
    rst <= '1';
    wait for clk_period;
    rst <= '0';
    wait;
end process;

-- Clock process definitions
clk_process :process
begin
    clk <= '0';
    wait for clk_period/2;
    clk <= '1';
    wait for clk_period/2;
end process;

end Behavioral;
