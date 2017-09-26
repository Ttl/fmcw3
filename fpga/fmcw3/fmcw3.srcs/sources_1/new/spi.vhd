----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 29.07.2017 13:27:56
-- Design Name: 
-- Module Name: spi - Behavioral
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

entity spi is
    Generic (SPI_CLK_DIVIDER : integer := 1);
Port ( clk : in STD_LOGIC;
       rst : in STD_LOGIC;
       data_in : in STD_LOGIC_VECTOR (31 downto 0);
       write : in STD_LOGIC;
       busy : out STD_LOGIC;
       spi_clk : out STD_LOGIC;
       spi_data : out STD_LOGIC;
       spi_le : out STD_LOGIC;
       ack : out STD_LOGIC);
end spi;

architecture Behavioral of spi is

signal spi_le_int : std_logic := '0';
signal spi_clk_en : std_logic := '0';
signal spi_clk_int : std_logic := '0';
signal spi_reg : std_logic_vector(31 downto 0);

constant LE_DELAY : integer := 4;
constant LE_LENGTHEN : integer := LE_DELAY-1;
constant BUSY_DELAY : integer := LE_DELAY + LE_LENGTHEN + 1;

-- LE lengthen
signal pipe_le : std_logic_vector(LE_LENGTHEN-1 downto 0) := (others => '0');
signal pipe_busy : std_logic_vector(BUSY_DELAY-1 downto 0) := (others => '0');

signal write_buffer : std_logic := '0';

begin

process(clk, rst, data_in, write)
variable count : unsigned(7 downto 0) := to_unsigned(0, 8);
variable done : std_logic := '1';
variable bits, bits_prev : unsigned(7 downto 0) := to_unsigned(0, 8);
variable delay : unsigned(7 downto 0) := to_unsigned(LE_DELAY, 8);
variable data_in_buffer : std_logic_vector(31 downto 0);

begin

if rst = '1' then
    delay := to_unsigned(LE_DELAY, 8);
    spi_le_int <= '0';
    spi_clk_en <= '0';
    spi_clk_int <= '0';
    done := '1';
    ack <= '0';
elsif rising_edge(clk) then
    spi_le_int <= '0';
    ack <= '0';
    
    if delay /= to_unsigned(0, 8) then
        delay := delay - 1;
    end if;

    -- Buffer write requests
    -- Write must begin on correct SPI clock phase
    if write = '1' and done = '1' and delay = to_unsigned(0, 8) then
        data_in_buffer := data_in;
        write_buffer <= '1';
        ack <= '1';
    end if;
    
    if done = '1' then
        spi_clk_en <= '0';
    else
        spi_clk_en <= '1';
    end if;

    if count = SPI_CLK_DIVIDER then
        count := to_unsigned(0, 8);
        if spi_clk_en = '1' then
            spi_clk_int <= not spi_clk_int;
        else
            spi_clk_int <= '0';
        end if;
        
        if spi_clk_int = '1' then
            -- Shift new bit
            if write_buffer = '0' then
                spi_reg <= spi_reg(30 downto 0)&'0';
            end if;
            
            bits_prev := bits;
            
            if bits /= to_unsigned(0, 8) then
                bits := bits - to_unsigned(1, 8);
            end if;
            
            if bits_prev = to_unsigned(1, 8) and bits = to_unsigned(0, 8) then
                spi_le_int <= '1';
                done := '1';
                delay := to_unsigned(LE_DELAY, 8);
            end if;
        end if;
            
        -- Start a new write on falling SPI clock edge
        if spi_clk_en = '0' or (spi_clk_en = '1' and spi_clk_int = '1') then
            if write_buffer = '1' then
                write_buffer <= '0';
                bits := to_unsigned(32, 8);
                spi_reg <= data_in_buffer;
                done := '0';
            end if;
        end if;
    else
        count := count + to_unsigned(1, 8);
    end if;

end if;
end process;

-- busy <= spi_clk_en;
spi_clk <= spi_clk_int;
spi_data <= spi_reg(31);


le_busy_process : process(clk, spi_le_int, spi_clk_en)
variable le_or : std_logic := '0';
variable busy_or : std_logic := '0';
begin
if rising_edge(clk) then
    pipe_le(0) <= spi_le_int;
    pipe_busy(0) <= spi_clk_en;
    
    le_or := '0';
    busy_or := '0';
    
    for i in 1 to LE_LENGTHEN - 1 loop
        pipe_le(i) <= pipe_le(i-1);
        le_or := le_or or pipe_le(i);
    end loop;
    
    for i in 1 to BUSY_DELAY - 1 loop
        pipe_busy(i) <= pipe_busy(i-1);
        busy_or := busy_or or pipe_busy(i);
    end loop;
    
    spi_le <= le_or;
    busy <= busy_or;
end if;

end process;

end Behavioral;
