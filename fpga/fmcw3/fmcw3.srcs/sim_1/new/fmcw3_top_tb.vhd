----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 14.08.2017 18:27:52
-- Design Name: 
-- Module Name: fmcw3_top_tb - Behavioral
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

entity fmcw3_top_tb is
--  Port ( );
end fmcw3_top_tb;

architecture Behavioral of fmcw3_top_tb is

signal clk : std_logic;

signal ft_clk : std_logic;
signal ft_data : std_logic_vector(7 downto 0);
signal ft_rd, ft_wr, ft_oe, ft_rxf, ft_txe, ft_suspend, ft_siwua : std_logic := '0';

constant clk_period : time := 25 ns;
constant ft_clk_period : time := 16 ns;

signal sd_data : std_logic_vector(3 downto 0) := (others => 'Z');
signal adf_clk, adf_le, adf_ce, sd_cmd, sd_clk, sd_detect, mix_enbl, led, pa_off, adf_muxout, adf_txdata, adf_done, adf_data : std_logic := '0';
signal ext1, ext2 : std_logic_vector(5 downto 0);
signal spi_cs, spi_din, spi_dout : std_logic := '0';
signal adc_d : std_logic_vector(11 downto 0) := (others => '0');
signal adc_of, adc_oe, adc_shdn : std_logic_vector(1 downto 0) := (others => '0');

--constant PACKET_LENGTH : integer := 8;
--type memory_type is array (0 to PACKET_LENGTH-1) of std_logic_vector(7 downto 0);
--signal memory : memory_type := (
--0 => "10101010",
--1 => "00000001",
--2 => "00000001",
--3 => "11111111",
--4 => "10101010",
--5 => "00000001",
--6 => "00000000",
--7 => "11111111",
--others => (others => '0'));
constant PACKET_LENGTH : integer := 25;
type memory_type is array (0 to PACKET_LENGTH-1) of std_logic_vector(7 downto 0);
signal memory : memory_type := (
0 => "10101010",
1 => std_logic_vector(to_unsigned(4, 8)),
2 => std_logic_vector(to_unsigned(4, 8)),
3 => "11111111",
4 => "10101010",
5 => "00000000",
6 => "00000011",

7 => "10101010",
8 => std_logic_vector(to_unsigned(4, 8)),
9 => std_logic_vector(to_unsigned(4, 8)),
10 => "00000000",
11 => "11001100",
12 => "00110011",
13 => "00000001",

14 => "10101010",
15 => std_logic_vector(to_unsigned(1, 8)),
16 => std_logic_vector(to_unsigned(7, 8)),
17 => std_logic_vector(to_unsigned(1, 8)),

18 => "10101010",
19 => std_logic_vector(to_unsigned(4, 8)),
20 => std_logic_vector(to_unsigned(6, 8)),
21 => std_logic_vector(to_unsigned(100, 8)),
22 => std_logic_vector(to_unsigned(0, 8)),
23 => std_logic_vector(to_unsigned(0, 8)),
24 => std_logic_vector(to_unsigned(0, 8)),
others => (others => '0'));

signal ft_dataout : std_logic_vector(7 downto 0);

begin

--ft_rxf <= '1';
--ft_txe <= '0';

top : entity work.fmcw3_top
    Port map ( clk => clk,
           ft_data => ft_data,
           ft_rxf => ft_rxf,
           ft_txe => ft_txe,
           ft_rd => ft_rd,
           ft_wr => ft_wr,
           ft_siwua => ft_siwua,
           ft_clkout => ft_clk,
           ft_oe => ft_oe,
           ft_suspend => ft_suspend,
           adc_d => adc_d,
           adc_of => adc_of,
           adc_oe => adc_oe,
           adc_shdn => adc_shdn,
           sd_data => sd_data,
           sd_cmd => sd_cmd,
           sd_clk => sd_clk,
           sd_detect => sd_detect,
           mix_enbl => mix_enbl,
           led => led,
           pa_off => pa_off,
           adf_ce => adf_ce,
           adf_le => adf_le,
           adf_clk => adf_clk,
           adf_muxout => adf_muxout,
           adf_txdata => adf_txdata,
           adf_data => adf_data,
           adf_done => adf_done,
           ext1 => ext1,
           ext2 => ext2,
           spi_cs => spi_cs,
           spi_din => spi_din,
           spi_dout => spi_dout);

data_process : process(clk)
begin
    if rising_edge(clk) or falling_edge(clk) then
        adf_muxout <= '0';
        if unsigned(adc_d) < to_unsigned(4, 12) then
            adf_muxout <= '1';
        end if;
        adc_d <= std_logic_vector(unsigned(adc_d) + 1);
    end if;
end process;

-- Clock process definitions
clk_process : process
begin
    clk <= '0';
    wait for clk_period/2;
    clk <= '1';
    wait for clk_period/2;
end process;

ft_clk_process : process
begin
    ft_clk <= '0';
    wait for ft_clk_period/2;
    ft_clk <= '1';
    wait for ft_clk_period/2;
end process;

comm_process : process(ft_clk, ft_rd)
variable data_count : integer := 0;
variable done : std_logic := '0';
begin

if rising_edge(ft_clk) then
    if done = '0' then
        ft_rxf <= '0';
    else
        ft_rxf <= '1';
    end if;
    
    if ft_rd = '0' then
        if data_count = PACKET_LENGTH-1 then
            data_count := 0;
            done := '1';
        else
            data_count := data_count + 1;
        end if;
    end if;
    ft_dataout <= memory(data_count);
end if;

end process;

ft_data <= ft_dataout when ft_oe = '0' else (others => 'Z');

end Behavioral;
