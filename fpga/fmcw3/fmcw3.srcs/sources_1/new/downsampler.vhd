----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 23.08.2017 16:47:11
-- Design Name: 
-- Module Name: downsampler - Behavioral
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

entity downsampler is
    Port ( clk : in STD_LOGIC;
           enable : in STD_LOGIC;
           data_a_in : in STD_LOGIC_VECTOR (15 downto 0);
           data_b_in : in STD_LOGIC_VECTOR (15 downto 0);
           data_a_out : out STD_LOGIC_VECTOR (15 downto 0);
           data_b_out : out STD_LOGIC_VECTOR (15 downto 0);
           data_valid_in : in STD_LOGIC;
           data_valid_out : out STD_LOGIC;
           sample_valid_in : in STD_LOGIC;
           sample_valid_out : out STD_LOGIC);
end downsampler;

architecture Behavioral of downsampler is

constant generate_fir : boolean := true;

-- R*(N-1)/2, where N = number of taps, R = Input data rate
constant VALID_DELAY : integer := 460;

COMPONENT fir_compiler_1
  PORT (
    aclk : IN STD_LOGIC;
    s_axis_data_tvalid : IN STD_LOGIC;
    s_axis_data_tready : OUT STD_LOGIC;
    s_axis_data_tdata : IN STD_LOGIC_VECTOR(31 DOWNTO 0);
    m_axis_data_tvalid : OUT STD_LOGIC;
    m_axis_data_tdata : OUT STD_LOGIC_VECTOR(31 DOWNTO 0)
  );
END COMPONENT;

signal data_in : std_logic_vector(31 downto 0);
signal data_out : std_logic_vector(31 downto 0);
signal valid_pipe : std_logic_vector(VALID_DELAY-1 downto 0) := (others=> '0');

begin

g_fir : if generate_fir generate

fir1 : fir_compiler_1
  PORT MAP (
    aclk => clk,
    s_axis_data_tvalid => data_valid_in,
    s_axis_data_tready => open,
    s_axis_data_tdata => data_in,
    m_axis_data_tvalid => data_valid_out,
    m_axis_data_tdata => data_out
  );

end generate;

g_not_fir : if not generate_fir generate

    data_out(39 downto 25) <= data_in(31 downto 16);
    data_out(15 downto 0) <= data_in(15 downto 0);
    
    process(clk)
    variable count : unsigned(7 downto 0) := (others => '0');
    begin
    
        if rising_edge(clk) then
            if count = to_unsigned(1, 8) then
                count := (others => '0');
                data_valid_out <= '1';
            else
                count := count + 1;
                data_valid_out <= '0';
            end if;
        end if;
    end process;
    
end generate;

data_in <= data_a_in&data_b_in;

data_a_out <= data_out(31 downto 16) when enable = '1' else data_a_in;
data_b_out <= data_out(15 downto 0) when enable = '1' else data_a_in;
sample_valid_out <= valid_pipe(VALID_DELAY-1) when enable = '1' else sample_valid_in;


delay_process : process(clk)
begin

if rising_edge(clk) then

    valid_pipe(0) <= sample_valid_in;
    
    for i in 1 to VALID_DELAY - 1 loop
        valid_pipe(i) <= valid_pipe(i-1);
    end loop;
    
end if;

end process;

end Behavioral;
