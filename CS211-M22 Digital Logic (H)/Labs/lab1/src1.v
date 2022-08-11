`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2021/01/14 16:40:35
// Design Name: 
// Module Name: src1
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


module src1(
    input [15:0] sw,
    output [15:0] led
    );
    assign led = sw;
    //assign led = ~sw;
endmodule
