`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2020/07/29 18:49:25
// Design Name: 
// Module Name: song
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module song(
	input sys_clk,									//输入时钟100MHz，管脚绑定P17
	output reg speaker,								//输出至扬声器的信号，本例中为方波，管脚绑定T1
	output reg [2:0] cs,							//数码管片选信号
	output [6:0] seg_7s								//用数码管显示音符
);
	wire clk_6mhz;									//用于产生各种音阶频率的基准频率
	clk_self  #(8)  u1(
						.clk(sys_clk),
						.clk_self(clk_6mhz)			//6.25MHz时钟信号
						);
	
	wire clk_4hz;									//用于控制音长（节拍）的时钟频率
	clk_self  #(12500000)	u2(
								.clk(sys_clk),
								.clk_self(clk_4hz)	//得到4Hz时钟信号
								);
								
	reg [13:0] divider,origin;
	reg carry;
	reg [7:0] counter;
	reg [3:0] high,med,low,num;
	
	always @(posedge clk_6mhz)						//通过置数，改变分频比
	begin
		if(divider == 16383)
		begin
			divider <= origin;
			carry <= 1;
		end
		else begin
			divider <= divider + 1;
			carry <= 0;
		end
	end
	
	always @(posedge carry)							//2分频得到方波信号
	begin
		speaker <= ~speaker;
	end
	
	always @(posedge clk_4hz)						//根据不同的音符，预置分频比
	begin
		case({high,med,low})
			'h001:	origin <= 4915;
			'h002:	origin <= 6168;
			'h003:	origin <= 7281;
			'h004:	origin <= 7792;
			'h005:	origin <= 8730;
			'h006:	origin <= 9565;
			'h007:	origin <= 10310;
			'h010:	origin <= 10647;
			'h020:	origin <= 11272;
			'h030:	origin <= 11831;
			'h040:	origin <= 12094;
			'h050:	origin <= 12556;
			'h060:	origin <= 12947;
			'h070:	origin <= 13346;
			'h100:	origin <= 13516;
			'h200:	origin <= 13829;
			'h300:	origin <= 14109;
			'h400:	origin <= 14235;
			'h500:	origin <= 14470;
			'h600:	origin <= 14678;
			'h700:	origin <= 14864;
			'h000:	origin <= 16383;
		endcase
	end
	
	always @(posedge clk_4hz)			//计时，以实现循环演奏
	begin
		if(counter == 134)
			counter <= 0;
		else
			counter <= counter + 1;
		case(counter)
			0:begin		{high,med,low} <= 'h003;	cs <= 3'b***;	end		//低音
			1:begin		{high,med,low} <= 'h003;	cs <= 3'b***;	end		//持续4个节拍
			2:begin		{high,med,low} <= 'h003;	cs <= 3'b***;	end
			3:begin		{high,med,low} <= 'h003;	cs <= 3'b***;	end
			4:begin		{high,med,low} <= 'h005;	cs <= 3'b***;	end		//低音5
			5:begin		{high,med,low} <= 'h005;	cs <= 3'b***;	end		//持续3个节拍
			6:begin		{high,med,low} <= 'h005;	cs <= 3'b***;	end
			7:begin		{high,med,low} <= 'h006;	cs <= 3'b***;	end		//低音6
			8:begin		{high,med,low} <= 'h010;	cs <= 3'b***;	end		//中音1
			9:begin		{high,med,low} <= 'h010;	cs <= 3'b***;	end		//持续3个节拍
			10:begin	{high,med,low} <= 'h010;	cs <= 3'b***;	end
			11:begin	{high,med,low} <= 'h020;	cs <= 3'b***;	end		//中音2
			12:begin	{high,med,low} <= 'h006;	cs <= 3'b***;	end		//低音6
			13:begin	{high,med,low} <= 'h010;	cs <= 3'b***;	end
			14:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			15:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			16:begin	{high,med,low} <= 'h050;	cs <= 3'b***;	end		//中音5
			17:begin	{high,med,low} <= 'h050;	cs <= 3'b***;	end
			18:begin	{high,med,low} <= 'h050;	cs <= 3'b***;	end
			19:begin	{high,med,low} <= 'h100;	cs <= 3'b***;	end		//高音1
			20:begin	{high,med,low} <= 'h060;	cs <= 3'b***;	end
			21:begin	{high,med,low} <= 'h050;	cs <= 3'b***;	end
			22:begin	{high,med,low} <= 'h030;	cs <= 3'b***;	end
			23:begin	{high,med,low} <= 'h050;	cs <= 3'b***;	end
			24:begin	{high,med,low} <= 'h020;	cs <= 3'b***;	end
			25:begin	{high,med,low} <= 'h020;	cs <= 3'b***;	end
			26:begin	{high,med,low} <= 'h020;	cs <= 3'b***;	end
			27:begin	{high,med,low} <= 'h020;	cs <= 3'b***;	end
			28:begin	{high,med,low} <= 'h020;	cs <= 3'b***;	end
			29:begin	{high,med,low} <= 'h020;	cs <= 3'b***;	end
			30:begin	{high,med,low} <= 'h000;	cs <= 3'b***;	end
			31:begin	{high,med,low} <= 'h000;	cs <= 3'b***;	end
			32:begin	{high,med,low} <= 'h020;	cs <= 3'b***;	end
			33:begin	{high,med,low} <= 'h020;	cs <= 3'b***;	end
			34:begin	{high,med,low} <= 'h020;	cs <= 3'b***;	end
			35:begin	{high,med,low} <= 'h030;	cs <= 3'b***;	end
			36:begin	{high,med,low} <= 'h007;	cs <= 3'b***;	end
			37:begin	{high,med,low} <= 'h007;	cs <= 3'b***;	end
			38:begin	{high,med,low} <= 'h006;	cs <= 3'b***;	end
			39:begin	{high,med,low} <= 'h006;	cs <= 3'b***;	end
			40:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			41:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			42:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			43:begin	{high,med,low} <= 'h006;	cs <= 3'b***;	end
			44:begin	{high,med,low} <= 'h010;	cs <= 3'b***;	end
			45:begin	{high,med,low} <= 'h010;	cs <= 3'b***;	end
			46:begin	{high,med,low} <= 'h020;	cs <= 3'b***;	end
			47:begin	{high,med,low} <= 'h020;	cs <= 3'b***;	end
			48:begin	{high,med,low} <= 'h003;	cs <= 3'b***;	end
			49:begin	{high,med,low} <= 'h003;	cs <= 3'b***;	end
			50:begin	{high,med,low} <= 'h010;	cs <= 3'b***;	end
			51:begin	{high,med,low} <= 'h010;	cs <= 3'b***;	end
			52:begin	{high,med,low} <= 'h006;	cs <= 3'b***;	end
			53:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			54:begin	{high,med,low} <= 'h006;	cs <= 3'b***;	end
			55:begin	{high,med,low} <= 'h010;	cs <= 3'b***;	end
			56:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			57:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			58:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			59:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			60:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			61:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			62:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			63:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			64:begin	{high,med,low} <= 'h030;	cs <= 3'b***;	end
			65:begin	{high,med,low} <= 'h030;	cs <= 3'b***;	end
			66:begin	{high,med,low} <= 'h030;	cs <= 3'b***;	end
			67:begin	{high,med,low} <= 'h050;	cs <= 3'b***;	end
			68:begin	{high,med,low} <= 'h007;	cs <= 3'b***;	end
			69:begin	{high,med,low} <= 'h007;	cs <= 3'b***;	end
			70:begin	{high,med,low} <= 'h020;	cs <= 3'b***;	end
			71:begin	{high,med,low} <= 'h020;	cs <= 3'b***;	end
			72:begin	{high,med,low} <= 'h006;	cs <= 3'b***;	end
			73:begin	{high,med,low} <= 'h010;	cs <= 3'b***;	end
			74:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			75:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			76:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			77:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			78:begin	{high,med,low} <= 'h000;	cs <= 3'b***;	end
			79:begin	{high,med,low} <= 'h000;	cs <= 3'b***;	end
			80:begin	{high,med,low} <= 'h003;	cs <= 3'b***;	end
			81:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			82:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			83:begin	{high,med,low} <= 'h003;	cs <= 3'b***;	end
			84:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			85:begin	{high,med,low} <= 'h006;	cs <= 3'b***;	end
			86:begin	{high,med,low} <= 'h007;	cs <= 3'b***;	end
			87:begin	{high,med,low} <= 'h020;	cs <= 3'b***;	end
			88:begin	{high,med,low} <= 'h006;	cs <= 3'b***;	end
			89:begin	{high,med,low} <= 'h006;	cs <= 3'b***;	end
			90:begin	{high,med,low} <= 'h006;	cs <= 3'b***;	end
			91:begin	{high,med,low} <= 'h006;	cs <= 3'b***;	end
			92:begin	{high,med,low} <= 'h006;	cs <= 3'b***;	end
			93:begin	{high,med,low} <= 'h006;	cs <= 3'b***;	end
			94:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			95:begin	{high,med,low} <= 'h006;	cs <= 3'b***;	end
			96:begin	{high,med,low} <= 'h010;	cs <= 3'b***;	end
			97:begin	{high,med,low} <= 'h010;	cs <= 3'b***;	end
			98:begin	{high,med,low} <= 'h010;	cs <= 3'b***;	end
			99:begin	{high,med,low} <= 'h020;	cs <= 3'b***;	end
			100:begin	{high,med,low} <= 'h050;	cs <= 3'b***;	end
			101:begin	{high,med,low} <= 'h050;	cs <= 3'b***;	end
			102:begin	{high,med,low} <= 'h030;	cs <= 3'b***;	end
			103:begin	{high,med,low} <= 'h030;	cs <= 3'b***;	end
			104:begin	{high,med,low} <= 'h020;	cs <= 3'b***;	end
			105:begin	{high,med,low} <= 'h020;	cs <= 3'b***;	end
			106:begin	{high,med,low} <= 'h030;	cs <= 3'b***;	end
			107:begin	{high,med,low} <= 'h020;	cs <= 3'b***;	end
			108:begin	{high,med,low} <= 'h010;	cs <= 3'b***;	end
			109:begin	{high,med,low} <= 'h010;	cs <= 3'b***;	end
			110:begin	{high,med,low} <= 'h006;	cs <= 3'b***;	end
			111:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			112:begin	{high,med,low} <= 'h003;	cs <= 3'b***;	end
			113:begin	{high,med,low} <= 'h003;	cs <= 3'b***;	end
			114:begin	{high,med,low} <= 'h003;	cs <= 3'b***;	end
			115:begin	{high,med,low} <= 'h003;	cs <= 3'b***;	end
			116:begin	{high,med,low} <= 'h010;	cs <= 3'b***;	end
			117:begin	{high,med,low} <= 'h010;	cs <= 3'b***;	end
			118:begin	{high,med,low} <= 'h010;	cs <= 3'b***;	end
			119:begin	{high,med,low} <= 'h010;	cs <= 3'b***;	end
			120:begin	{high,med,low} <= 'h006;	cs <= 3'b***;	end
			121:begin	{high,med,low} <= 'h010;	cs <= 3'b***;	end
			122:begin	{high,med,low} <= 'h006;	cs <= 3'b***;	end
			123:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			124:begin	{high,med,low} <= 'h003;	cs <= 3'b***;	end
			125:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			126:begin	{high,med,low} <= 'h006;	cs <= 3'b***;	end
			127:begin	{high,med,low} <= 'h010;	cs <= 3'b***;	end
			128:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			129:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			130:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			131:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			132:begin	{high,med,low} <= 'h005;	cs <= 3'b***;	end
			133:begin	{high,med,low} <= 'h000;	cs <= 3'b***;	end
			134:begin	{high,med,low} <= 'h000;	cs <= 3'b***;	end
			default:begin	{high,med,low} <= 'h000;	cs <= 3'b***;	end
		endcase
	end
	
	always @(*)
	begin
		case(cs)
			'b001:	num <= low;
			'b010:	num <= med;
			'b100:	num <= high;
			default:num <= 4'b0000;
		endcase
	end
	
	seg7 u3(
			.hex(num),
			.a_to_g(seg_7s)
	);
	
endmodule
