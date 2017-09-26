-- FT2232H USB Device Core
-- Operates in FT245 Style Synchronous FIFO Mode for high speed data transfers
-- Designer: Wes Pope
-- License: Public Domain
 
 
library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
 
entity usb_sync is 
	port (
		-- Bus signals
		signal clk : in std_logic;
		signal reset_n : in std_logic;
		signal read_n : in std_logic;
		signal write_n : in std_logic;
		signal chipselect : in std_logic;
		signal readdata : out std_logic_vector (7 downto 0);
		signal writedata : in std_logic_vector (7 downto 0);
		signal tx_full : out std_logic;
		signal rx_empty : out std_logic;
 
		-- FT2232 Bus Signals
		signal usb_clock : in std_logic;
		signal usb_data : inout std_logic_vector(7 downto 0);
		signal usb_rd_n : out std_logic;
		signal usb_wr_n : out std_logic;
		signal usb_oe_n : out std_logic;
		signal usb_rxf_n : in std_logic;
		signal usb_txe_n : in std_logic
		);
end entity usb_sync;
 
 
architecture rtl of usb_sync is
 
	signal rd_sig : std_logic;
	signal wr_sig : std_logic;
 
	signal rx_fifo_rddone : std_logic := '0';	
 
	signal rx_fifo_wrclk : std_logic;
	signal rx_fifo_rdreq : std_logic;
	signal rx_fifo_rdclk : std_logic;
	signal rx_fifo_wrreq : std_logic;
	signal rx_fifo_data : std_logic_vector(7 downto 0);
	signal rx_fifo_wrfull : std_logic;
	signal rx_fifo_q : std_logic_vector(7 downto 0);
	signal tx_fifo_wrfull, rx_fifo_rdempty : std_logic;
	signal rx_empty_int, rx_empty_prev : std_logic;
 
	signal tx_fifo_wrclk : std_logic;
	signal tx_fifo_rdreq : std_logic;
	signal tx_fifo_rdclk : std_logic;
	signal tx_fifo_wrreq : std_logic;
	signal tx_fifo_data : std_logic_vector(7 downto 0);
	signal tx_fifo_rdempty : std_logic;
	signal tx_fifo_q : std_logic_vector(7 downto 0);
	signal tx_fifo_wrusedw : std_logic_vector(11 downto 0);
 
	signal ft2232_wait : integer range 0 to 1 := 0;
	signal ft2232_bus_oe_mode : integer range 0 to 3 := 0;
	signal ft2232_tx_fifo_read : std_logic;
	signal ft2232_rx_fifo_write : std_logic;
	signal ft2232_tx_please : std_logic;
	signal ft2232_rx_please : std_logic;
 
 
COMPONENT fifo_generator_0
      PORT (
        wr_clk : IN STD_LOGIC;
        rd_clk : IN STD_LOGIC;
        din : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
        wr_en : IN STD_LOGIC;
        rd_en : IN STD_LOGIC;
        dout : OUT STD_LOGIC_VECTOR(7 DOWNTO 0);
        full : OUT STD_LOGIC;
        empty : OUT STD_LOGIC
      );
    END COMPONENT;
 
begin
 
rx_dcfifo : fifo_generator_0
PORT MAP (
    wr_clk => rx_fifo_wrclk,
    rd_clk => rx_fifo_rdclk,
    din => rx_fifo_data,
    wr_en => rx_fifo_wrreq,
    rd_en => rx_fifo_rdreq,
    dout => rx_fifo_q,
    full => rx_fifo_wrfull,
    empty => rx_fifo_rdempty
);


tx_dcfifo : fifo_generator_0
PORT MAP (
    wr_clk => tx_fifo_wrclk,
    rd_clk => tx_fifo_rdclk,
    din => tx_fifo_data,
    wr_en => tx_fifo_wrreq,
    rd_en => tx_fifo_rdreq,
    dout => tx_fifo_q,
    full => tx_fifo_wrfull,
    empty => tx_fifo_rdempty
);
 
    tx_full <= tx_fifo_wrfull;
 
	-- USB2232 side
	rx_fifo_wrclk <= usb_clock;
	tx_fifo_rdclk <= usb_clock;
 
	ft2232_tx_please <= '1' when usb_txe_n = '0' and tx_fifo_rdempty = '0' and ft2232_wait >= 1 else '0';
	ft2232_rx_please <= '1' when usb_rxf_n = '0' and rx_fifo_wrfull = '0' else '0';
 
	ft2232_tx_fifo_read <= '1' when ft2232_tx_please = '1' else '0';
	ft2232_rx_fifo_write <= '1' when ft2232_bus_oe_mode > 1 and ft2232_rx_please = '1' and ft2232_tx_please = '0' else '0';
 
	tx_fifo_rdreq <= ft2232_tx_fifo_read;
	rx_fifo_wrreq <= ft2232_rx_fifo_write;
 
	usb_rd_n <= '0' when ft2232_rx_fifo_write = '1' else '1';
	usb_wr_n <= '0' when ft2232_tx_fifo_read = '1' else '1';
	usb_oe_n <= '0' when ft2232_bus_oe_mode > 0 else '1';
	usb_data <= tx_fifo_q when ft2232_bus_oe_mode = 0 else (others => 'Z');
	rx_fifo_data <= usb_data when ft2232_bus_oe_mode > 0 and usb_rxf_n = '0' else (others => '0');
 
 
	-- Handle FIFOs to USB2232 in synchronous mode
	process (usb_clock)
	begin
 
		if usb_clock'event and usb_clock = '1' then
 
			-- Bias TX over RX
			if (ft2232_tx_please = '1' or ft2232_rx_please = '0') then
 
				ft2232_bus_oe_mode <= 0;
 
				if (usb_txe_n = '0' and tx_fifo_rdempty = '0') then
					ft2232_wait <= ft2232_wait + 1;
				else
					ft2232_wait <= 0;
				end if;
 
			elsif (ft2232_rx_please = '1') then
 
				ft2232_wait <= 0;
 
				-- Handle bus turn-around. Negate OE (and for atleast 1 clock)
				if (ft2232_bus_oe_mode < 3) then		
					ft2232_bus_oe_mode <= ft2232_bus_oe_mode + 1;
				end if;
 
			end if;
 
		end if;		
 
	end process;
 
 
	-- Bus side
	rx_fifo_rdclk <= clk;
	tx_fifo_wrclk <= clk;
	
	readdata <= rx_fifo_q;
 
	wr_sig <= '1' when chipselect = '1' and write_n = '0' else '0';
	rd_sig <= '1' when chipselect = '1' and read_n = '0' else '0';
 
	-- Handle FIFOs to Bus
	process (clk, reset_n)
	begin
 
		if reset_n = '0' then
 
			rx_fifo_rddone <= '0';
 
		elsif rising_edge(clk) then			
            rx_empty_int <= '1';
            rx_empty_prev <= rx_empty_int;
            
            rx_fifo_rdreq <= '0';
            
			if rd_sig = '1' then
			    rx_empty_int <= rx_fifo_rdempty;
				rx_fifo_rdreq <= '1';
			end if;
 
			if wr_sig = '1' then
				-- write fifo
				tx_fifo_wrreq <= '1';
				tx_fifo_data <= writedata(7 downto 0);
			else
				tx_fifo_wrreq <= '0';
			end if;
 
		end if;
 
	end process;	
 
rx_empty <= '0' when rx_empty_int = '0' else '1';
 
end rtl;