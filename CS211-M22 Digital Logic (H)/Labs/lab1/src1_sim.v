`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2021/01/14 16:41:33
// Design Name: 
// Module Name: src1_sim
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


module src1_sim();

reg [15:0] sw = 16'h0000;
wire [15:0] led;

src1 usrc1(
    .sw(sw),
    .led(led)
);

/*
src1 usrc2(sw,led);
*/

initial begin
    #5 sw = 16'h1111;    
    #1000 $finish;
end

always #10 sw = sw + 1;
endmodule
