----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 20.08.2017 12:07:38
-- Design Name: 
-- Module Name: sample_packer - Behavioral
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

entity sample_packer is
    Port ( clk : in STD_LOGIC;
           rst : in STD_LOGIC;
           data_a : in STD_LOGIC_VECTOR (15 downto 0);
           data_b : in STD_LOGIC_VECTOR (15 downto 0);
           adc_valid : in STD_LOGIC;
           sample_valid : in STD_LOGIC;
           usb_write : out STD_LOGIC;
           usb_writedata : out STD_LOGIC_VECTOR (7 downto 0);
           enable_a : in STD_LOGIC;
           enable_b : in STD_LOGIC;
           tx_full : in STD_LOGIC;
           clear_buffer : in STD_LOGIC);
end sample_packer;

architecture Behavioral of sample_packer is

constant PACKET_START : std_logic_vector(7 downto 0) := "01111111";

constant MEMORY_LENGTH : integer := 131072;
type memory_type is array (0 to MEMORY_LENGTH-1) of std_logic_vector(7 downto 0);
signal memory : memory_type := (others => (others => '0'));
signal write_pointer, read_pointer : unsigned(15 downto 0) := (others => '0');

signal sweep_count : std_logic_vector(7 downto 0) := (others => '0');
signal sample_valid_prev : std_logic := '0';
signal write_length : std_logic := '0';

signal data_a_int, data_b_int : std_logic_vector(15 downto 0) := (others => '0');
signal data_int_valid : std_logic := '0';

signal write_header : std_logic := '0';

begin

process(clk, rst)

type sample_state_type is (S_A1, S_A2, S_B1, S_B2);
variable sample_state : sample_state_type := S_A1;
variable wrote_memory : std_logic := '0';

begin

if rst = '1' then
    sweep_count <= (others => '0');
    write_length <= '0';
    data_int_valid <= '0';
    
elsif rising_edge(clk) then

    sample_valid_prev <= sample_valid;
    write_header <= '0';
    write_length <= '0';
    wrote_memory := '0';
    
    usb_writedata <= memory(to_integer(read_pointer));
    usb_write <= '0';
    
    if adc_valid = '1' and sample_valid = '1' then
        data_a_int <= data_a;
        data_b_int <= data_b;
        data_int_valid <= '1';
        if enable_a = '1' then
            sample_state := S_A1;
        else
            sample_state := S_B1;
        end if;
    end if;
    
    if enable_a = '1' or enable_b = '1' then
    
        if sample_valid = '1' and sample_valid_prev = '0' then
            write_header <= '1';
        end if;
        
        -- Make sure that header is not inserted between the samples
        if data_int_valid = '0' and write_header = '1' then
            -- Start of new sweep
            memory(to_integer(write_pointer)) <= PACKET_START;
            wrote_memory := '1';
            write_length <= '1';
        elsif data_int_valid = '0' and write_length = '1' then
            memory(to_integer(write_pointer)) <= sweep_count;
            sweep_count <= std_logic_vector(unsigned(sweep_count) + 1);
            wrote_memory := '1';
            write_length <= '0';
        else
            if data_int_valid = '1' then
                wrote_memory := '1';
                
                case sample_state is
                    
                    when S_A1 =>
                        memory(to_integer(write_pointer)) <= data_a_int(7 downto 0);
                        sample_state := S_A2;
                        
                    when S_A2 =>
                        memory(to_integer(write_pointer)) <= data_a_int(15 downto 8);
                        if enable_b = '1' then
                            sample_state := S_B1;
                        else
                            sample_state := S_A1;
                            data_int_valid <= '0';
                        end if;
                                        
                    when S_B1 =>
                        memory(to_integer(write_pointer)) <= data_b_int(7 downto 0);
                        sample_state := S_B2;
                        
                    when S_B2 =>
                        memory(to_integer(write_pointer)) <= data_b_int(15 downto 8);
                        sample_state := S_A1;
                        data_int_valid <= '0';
                        
                    when others =>
                        sample_state := S_A1;
                        data_int_valid <= '0';
                        
                end case;
            
            end if;
        end if;
    end if;

    if wrote_memory = '1' then
        if write_pointer = MEMORY_LENGTH - 1 then
            write_pointer <= (others => '0');
        else
            write_pointer <= write_pointer + 1;
        end if;
    end if;

    if read_pointer /= write_pointer and tx_full = '0' then
        usb_write <= '1';
        if read_pointer = MEMORY_LENGTH - 1 then
            read_pointer <= (others => '0');
        else
            read_pointer <= read_pointer + 1;
        end if;
    end if;
    
    if clear_buffer = '1' then
        write_pointer <= (others => '0');
        read_pointer <= (others => '0');
    end if;
    
end if;

end process;

end Behavioral;
