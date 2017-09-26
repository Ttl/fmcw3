----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 21.07.2017 08:50:31
-- Design Name: 
-- Module Name: fmcw3_top - Behavioral
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

entity fmcw3_top is
    Port ( clk : in STD_LOGIC;
           ft_data : inout STD_LOGIC_VECTOR (7 downto 0);
           ft_rxf : in STD_LOGIC;
           ft_txe : in STD_LOGIC;
           ft_rd : out STD_LOGIC;
           ft_wr : out STD_LOGIC;
           ft_siwua : out STD_LOGIC;
           ft_clkout : in STD_LOGIC;
           ft_oe : out STD_LOGIC;
           ft_suspend : in STD_LOGIC;
           adc_d : in STD_LOGIC_VECTOR (11 downto 0);
           adc_of : in STD_LOGIC_VECTOR (1 downto 0);
           adc_oe : out STD_LOGIC_VECTOR (1 downto 0);
           adc_shdn : out STD_LOGIC_VECTOR (1 downto 0);
           sd_data : inout STD_LOGIC_VECTOR (3 downto 0);
           sd_cmd : inout STD_LOGIC;
           sd_clk : out STD_LOGIC;
           sd_detect : in STD_LOGIC;
           mix_enbl : out STD_LOGIC;
           led : out STD_LOGIC;
           pa_off : out STD_LOGIC;
           adf_ce : out STD_LOGIC;
           adf_le : out STD_LOGIC;
           adf_clk : out STD_LOGIC;
           adf_muxout : in STD_LOGIC;
           adf_txdata : out STD_LOGIC;
           adf_data : out STD_LOGIC;
           adf_done : in STD_LOGIC;
           ext1 : out STD_LOGIC_VECTOR(5 downto 0);
           ext2 : out STD_LOGIC_VECTOR(5 downto 0);
           spi_cs : out STD_LOGIC;
           spi_din : in STD_LOGIC;
           spi_dout : out STD_LOGIC);
end fmcw3_top;

architecture Behavioral of fmcw3_top is

signal data_a, data_b : std_logic_vector(15 downto 0) := (others => '0');
signal adc_valid : std_logic;
signal rst : std_logic := '0';
signal rst_n : std_logic := '1';

signal read_n, write_n, chipselect : std_logic;
signal readdata, writedata : std_logic_vector(7 downto 0) := (others => '0');

signal tx_full, rx_empty : std_logic;
signal control_ready, readdata_valid : std_logic;

signal adf_writedata : std_logic_vector(31 downto 0);
signal adf_write, adf_busy : std_logic;

signal sample_valid, write_timer, write_delay, write_decimate, write_pa_off : std_logic;
signal timer_data : std_logic_vector(31 downto 0);

signal usb_write : std_logic;
signal enable_a, enable_b : std_logic;

signal data_a_downsampled, data_b_downsampled : std_logic_vector(15 downto 0);
signal enable_downsampler : std_logic;
signal adc_valid_downsampled, sample_valid_downsampled : std_logic;

signal pa_off_timer, pa_off_control : std_logic;
signal clear_buffer : std_logic;

begin

rst <= '0';
rst_n <= not rst;

write_n <= not usb_write;
read_n <= not control_ready;

sd_clk <= '0';
sd_cmd <= 'Z';
sd_data <= (others => 'Z');

spi_cs <= 'Z';
spi_dout <= 'Z';

adf_txdata <= '0';

ft_siwua <= '1';

adc : entity work.adc
    Port map ( clk => clk,
           adc_data => adc_d,
           data_a => data_a,
           data_b => data_b,
           valid => adc_valid);

downsample : entity work.downsampler
    Port map ( clk => clk,
           enable => enable_downsampler,
           data_a_in => data_a,
           data_b_in => data_b,
           data_a_out => data_a_downsampled,
           data_b_out => data_b_downsampled,
           data_valid_in => adc_valid,
           data_valid_out => adc_valid_downsampled,
           sample_valid_in => sample_valid,
           sample_valid_out => sample_valid_downsampled);

sweep_timer : entity work.sweep_timer
       Port map ( clk => clk,
              rst => rst,
              adf_done => adf_muxout,
              sample_valid => sample_valid,
              write_timer => write_timer,
              write_data => timer_data,
              write_delay => write_delay,
              write_decimate => write_decimate,
              write_pa_off => write_pa_off,
              pa_off => pa_off_timer);
              
sample_packer : entity work.sample_packer
      Port map ( clk => clk,
             rst => rst,
             data_a => data_a_downsampled,
             data_b => data_b_downsampled,
             adc_valid => adc_valid_downsampled,
             sample_valid => sample_valid_downsampled,
             usb_write => usb_write,
             usb_writedata => writedata,
             enable_a => enable_a,
             enable_b => enable_b,
             tx_full => tx_full,
             clear_buffer => clear_buffer);
           
usb_sync : entity work.usb_sync
    Port map(
       clk => clk,
       reset_n => rst_n,
       read_n => read_n,
       write_n => write_n,
       chipselect => '1',
       readdata => readdata,
       writedata => writedata,
       tx_full => tx_full,
       rx_empty => rx_empty,
       usb_clock => ft_clkout,
       usb_data => ft_data,
       usb_rd_n => ft_rd,
       usb_wr_n => ft_wr,
       usb_oe_n => ft_oe,
       usb_rxf_n => ft_rxf,
       usb_txe_n => ft_txe
       );
                   
control : entity work.control
    Port map ( clk => clk,
          rst => rst,
          data_in => readdata,
          ready => control_ready,
          data_valid => readdata_valid,
          led => led,
          pa_off => pa_off_control,
          mix_enbl => mix_enbl,
          adf_ce => adf_ce,
          adc_shdn => adc_shdn,
          adc_oe => adc_oe,
          adf_writedata => adf_writedata,
          adf_write => adf_write,
          spi_busy => adf_busy,
          write_timer => write_timer,
          write_delay => write_delay,
          write_decimate => write_decimate,
          timer_data => timer_data,
          enable_a => enable_a,
          enable_b => enable_b,
          enable_downsampler => enable_downsampler,
          clear_buffer => clear_buffer);
          
spi : entity work.spi
    Port map( clk => clk,
                 rst => rst,
                 data_in => adf_writedata,
                 write => adf_write,
                 busy => adf_busy,
                 spi_clk => adf_clk,
                 spi_data => adf_data,
                 spi_le => adf_le,
                 ack => open);
          
pa_off <= pa_off_timer or pa_off_control;

readdata_valid <= not rx_empty;

ext1(0) <= adf_muxout;
--ext1(5 downto 1) <= data_a(4 downto 0);
--ext2 <= data_b(5 downto 0);
ext1(5 downto 1) <= (others => 'Z');
ext2 <= (others => 'Z');

end Behavioral;
