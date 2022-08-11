`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2021/06/22 14:25:04
// Design Name: 
// Module Name: lab2_summer2021_sim4
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


module lab2_summer2021_sim4();
    reg sa, sb, sc;
    wire sz1, sz2, sz3;
   
    lab2_summer2021 uut4(sa, sb, sc, sz1, sz2, sz3);
   
   initial begin
       {sa, sb, sc} = 3'b000;
       while({sa, sb, sc} < 3'b111)
       begin
            #100 {sa, sb, sc} = {sa, sb, sc} +  1;
       end        
      #1000 $finish(1);
   end
endmodule
