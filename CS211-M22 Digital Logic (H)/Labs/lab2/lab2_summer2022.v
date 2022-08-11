`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2021/06/22 13:53:59
// Design Name: 
// Module Name: lab2_summer2021
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


module lab2_summer2021(
    input a,
    input b,
    input c,
    output z1,
    output z2,
    output z3
    );
    
    assign z1 = a | ((~b) & c);
    assign z2 = (a&b&c) | (a&b&(~c)) | (a&(~b)&c) | (a&(~b)&(~c)) | ((~a)&(~b)&c);//m7+m6+m5+m4+m1
    assign z3 = (a|b|c) & (a|(~b)|c) & (a|(~b)|(~c));//M0¡¤M2¡¤M3
endmodule
