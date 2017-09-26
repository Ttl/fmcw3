----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 12.08.2017 17:59:38
-- Design Name: 
-- Module Name: control - Behavioral
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

entity control is
    Port ( clk : in STD_LOGIC;
           rst : in STD_LOGIC;
           data_in : in STD_LOGIC_VECTOR(7 downto 0);
           ready : out STD_LOGIC;
           data_valid : in STD_LOGIC;
           led : out STD_LOGIC;
           pa_off : out STD_LOGIC;
           mix_enbl : out STD_LOGIC;
           adf_ce : out STD_LOGIC;
           adc_shdn : out STD_LOGIC_VECTOR(1 downto 0);
           adc_oe : out STD_LOGIC_VECTOR(1 downto 0);
           adf_writedata : out STD_LOGIC_VECTOR(31 downto 0);
           adf_write : out STD_LOGIC;
           spi_busy : in STD_LOGIC;
           write_timer : out STD_LOGIC;
           write_delay : out STD_LOGIC;
           write_decimate : out STD_LOGIC;
           write_pa_off : out STD_LOGIC;
           timer_data : out STD_LOGIC_VECTOR(31 downto 0);
           enable_a : out STD_LOGIC;
           enable_b : out STD_LOGIC;
           enable_downsampler : out STD_LOGIC;
           clear_buffer : out STD_LOGIC);
end control;

architecture Behavioral of control is

constant DATA_START : std_logic_vector(7 downto 0) := "10101010";

constant MAX_PACKET_LENGTH : integer := 16;
type memory_type is array (0 to MAX_PACKET_LENGTH-1) of std_logic_vector(7 downto 0);
signal memory : memory_type := (others => (others => '0'));

signal process_data : std_logic := '0';

signal led_int, mix_enbl_int : std_logic := '0';
signal pa_off_int, adf_ce_int : std_logic := '1';

signal adc_shdn_int : std_logic_vector(1 downto 0) := "11";
signal adc_oe_int : std_logic_vector(1 downto 0) := "10";

signal enable_a_int, enable_b_int : std_logic := '0';

signal enable_downsampler_int : std_logic := '1';

begin

process(clk, rst, data_in, data_valid)
type comm_state_type is (S_START, S_LENGTH, S_ID, S_DATA, S_PROCESS_DATA);
variable state : comm_state_type := S_START;
variable data_length : unsigned(7 downto 0) := to_unsigned(0, 8);
variable data_counter : unsigned(7 downto 0) := to_unsigned(0, 8);
variable data_id : unsigned(7 downto 0) := to_unsigned(0, 8);

begin

if rst = '1' then
    state := S_START; 
elsif rising_edge(clk) then

    ready <= '1';
    process_data <= '0';
    adf_write <= '0';
    adf_writedata <= (others => '-');
    timer_data <= (others => '-');
    write_timer <= '0';
    write_delay <= '0';
    write_decimate <= '0';
    write_pa_off <= '0';
    clear_buffer <= '0';
    
    case state is
        
        when S_START =>
            if data_valid = '1' then
                if data_in = DATA_START then
                    state := S_LENGTH;
                end if;
            end if;
            
        when S_LENGTH =>
            if data_valid = '1' then
                data_length := unsigned(data_in);
                data_counter := to_unsigned(0, 8);
                state := S_ID;
            end if;
            
        when S_ID =>
            if data_valid = '1' then
                data_id := unsigned(data_in);
                state := S_DATA;
            end if;    
            
        when S_DATA =>
            if data_valid = '1' then
                memory(to_integer(data_counter)) <= data_in;
                
                if data_counter /= MAX_PACKET_LENGTH-1 then
                    data_counter := data_counter + 1;
                end if;
                            
                if data_counter = data_length then
                    state := S_START;
                    process_data <= '1';
                end if;
            end if;
            
        when others =>
            state := S_START;
            
    end case;
    
    if process_data = '1' then
    
        ready <= '0';
    
        case to_integer(data_id) is
    
            when 0 =>
                if memory(0)(0) = '1' then
                    led_int <= '0';
                end if;
                if memory(0)(1) = '1' then
                    pa_off_int <= '0';
                end if;
                if memory(0)(2) = '1' then
                    mix_enbl_int <= '0';
                end if;
                if memory(0)(3) = '1' then
                    adf_ce_int <= '0';
                end if;
                
            when 1 =>
                if memory(0)(0) = '1' then
                    led_int <= '1';
                end if;
                if memory(0)(1) = '1' then
                    pa_off_int <= '1';
                end if;
                if memory(0)(2) = '1' then
                    mix_enbl_int <= '1';
                end if;
                if memory(0)(3) = '1' then
                    adf_ce_int <= '1';
                end if;
                
            when 2 =>
                if memory(0)(0) = '1' then
                    adc_oe_int(0) <= '0';
                end if;
                if memory(0)(1) = '1' then
                    adc_oe_int(1) <= '0';
                end if;
                if memory(0)(2) = '1' then
                    adc_shdn_int(0) <= '0';
                end if;
                if memory(0)(3) = '1' then
                    adc_shdn_int(1) <= '0';
                end if;
            
            when 3 =>
                if memory(0)(0) = '1' then
                    adc_oe_int(0) <= '1';
                end if;
                if memory(0)(1) = '1' then
                    adc_oe_int(1) <= '1';
                end if;
                if memory(0)(2) = '1' then
                    adc_shdn_int(0) <= '1';
                end if;
                if memory(0)(3) = '1' then
                    adc_shdn_int(1) <= '1';
                end if;
                
            when 4 =>
                if spi_busy = '1' then
                    process_data <= '1';
                    ready <= '0';
                elsif data_length = to_unsigned(4, 8) then
                    adf_writedata(7 downto 0) <= memory(0)(7 downto 0);
                    adf_writedata(15 downto 8) <= memory(1)(7 downto 0);
                    adf_writedata(23 downto 16) <= memory(2)(7 downto 0);
                    adf_writedata(31 downto 24) <= memory(3)(7 downto 0);
                    adf_write <= '1';
                end if;
                
            when 5 =>
                if data_length = to_unsigned(4, 8) then
                    timer_data(7 downto 0) <= memory(0)(7 downto 0);
                    timer_data(15 downto 8) <= memory(1)(7 downto 0);
                    timer_data(23 downto 16) <= memory(2)(7 downto 0);
                    timer_data(31 downto 24) <= memory(3)(7 downto 0);
                    write_timer <= '1';
                end if;
                
            when 6 =>
                if data_length = to_unsigned(4, 8) then
                    timer_data(7 downto 0) <= memory(0)(7 downto 0);
                    timer_data(15 downto 8) <= memory(1)(7 downto 0);
                    timer_data(23 downto 16) <= memory(2)(7 downto 0);
                    timer_data(31 downto 24) <= memory(3)(7 downto 0);
                    write_delay <= '1';
                end if;
                
            when 7 =>
                enable_a_int <= memory(0)(0);
                enable_b_int <= memory(0)(1);
                
            when 8 =>
                enable_downsampler_int <= memory(0)(0);
                
            when 9 =>
                if data_length = to_unsigned(2, 8) then
                    timer_data(7 downto 0) <= memory(0)(7 downto 0);
                    timer_data(15 downto 8) <= memory(1)(7 downto 0);
                    timer_data(31 downto 16) <= (others => '-');
                    write_decimate <= '1';
                end if;
                
            when 10 =>
                if data_length = to_unsigned(4, 8) then
                    timer_data(7 downto 0) <= memory(0)(7 downto 0);
                    timer_data(15 downto 8) <= memory(1)(7 downto 0);
                    timer_data(23 downto 16) <= memory(2)(7 downto 0);
                    timer_data(31 downto 24) <= memory(3)(7 downto 0);
                    write_pa_off <= '1';
                end if;
                
            when 11 =>
                clear_buffer <= '1';
                
            when others =>
                
        end case;
    
    end if;

end if;

end process;

led <= led_int;
pa_off <= pa_off_int;
mix_enbl <= mix_enbl_int;
adf_ce <= adf_ce_int;
adc_shdn <= adc_shdn_int;
adc_oe <= adc_oe_int;
enable_a <= enable_a_int;
enable_b <= enable_b_int;
enable_downsampler <= enable_downsampler_int;

end Behavioral;
