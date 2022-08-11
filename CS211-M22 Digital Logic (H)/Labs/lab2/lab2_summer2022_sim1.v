`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2021/06/22 13:55:15
// Design Name: 
// Module Name: lab2_summer2021_sim1
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


module lab2_summer2021_sim1();
    reg a_sim, b_sim, c_sim;
    wire z1_sim, z2_sim, z3_sim;
    
    lab2_summer2021 uut1(
        .a(a_sim),
        .b(b_sim),
        .c(c_sim),
        .z1(z1_sim),
        .z2(z2_sim),
        .z3(z3_sim)
    );
    
    initial begin
        {a_sim, b_sim, c_sim} = 3'b000;
        #1000 $finish;
    end
    
    always #10 {a_sim, b_sim, c_sim} = {a_sim, b_sim, c_sim} + 3'b001;
endmodule
