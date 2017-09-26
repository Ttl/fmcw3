----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 30.07.2017 09:25:24
-- Design Name: 
-- Module Name: usb_sync_tb - Behavioral
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

entity usb_sync_tb is
--  Port ( );
end usb_sync_tb;

architecture Behavioral of usb_sync_tb is

signal clk, rst, rst_n : std_logic;

signal usb_clk : std_logic;
signal usb_data, usb_txdata : std_logic_vector(7 downto 0) := (others => '0');
signal usb_rd_n, usb_wr_n, usb_oe_n, usb_rxf_n, usb_txe_n : std_logic := '0';

signal read_n, write_n, chipselect : std_logic;
signal readdata, writedata : std_logic_vector(7 downto 0) := (others => '0');

signal tx_full, rx_empty : std_logic;

constant clk_period : time := 25 ns;
constant usb_clk_period : time := 16 ns;

signal bus_read : std_logic_vector(7 downto 0);

begin

chipselect <= '1';
write_n <= '1';
read_n <= '0';
rst_n <= not rst;

txdata_counter : process(usb_clk)
begin
if rising_edge(usb_clk) then
    usb_txdata <= std_logic_vector(unsigned(usb_txdata) + 1);
end if;
end process;


oe_process : process
begin

    wait until usb_clk'event;
    if usb_oe_n = '1' then
        -- Disable output
        wait for 7 ns;
        usb_data <= (others => 'Z');
    else
        wait for 7 ns;
        usb_data <= usb_txdata;
    end if;

end process;

read_process : process
begin

usb_rxf_n <= '1';
wait for 10*usb_clk_period;
wait until rising_edge(usb_clk);
usb_rxf_n <= '0';
wait for 10*usb_clk_period;

end process;

usb_sync : entity work.usb_sync
	Port map(
		clk => clk,
		reset_n => rst_n,
        read_n => read_n,
        write_n => write_n,
        chipselect => chipselect,
        readdata => readdata,
        writedata => writedata,
        tx_full => tx_full,
        rx_empty => rx_empty,
		usb_clock => usb_clk,
		usb_data => usb_data,
		usb_rd_n => usb_rd_n,
		usb_wr_n => usb_wr_n,
		usb_oe_n => usb_oe_n,
		usb_rxf_n => usb_rxf_n,
		usb_txe_n => usb_txe_n
		);

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

usb_clk_process : process
begin
    usb_clk <= '0';
    wait for usb_clk_period/2;
    usb_clk <= '1';
    wait for usb_clk_period/2;
end process;

bus_process : process(clk, rx_empty)

begin

    if rising_edge(clk) then
        if rx_empty = '0' then
            bus_read <= readdata;
        else
            bus_read <= (others => 'U');
        end if;
    end if;

end process;

end Behavioral;
