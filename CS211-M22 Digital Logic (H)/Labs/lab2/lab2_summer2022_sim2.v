`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2021/06/22 13:59:35
// Design Name: 
// Module Name: lab2_summer2021_sim2
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


module lab2_summer2021_sim2();
    reg sa, sb, sc;
    wire sz1, sz2, sz3;
   
    lab2_summer2021 uut2(sa, sb, sc, sz1, sz2, sz3);
   
   initial begin
       {sa, sb, sc} = 3'b000;
       repeat(7)
       begin
            #100 {sa, sb, sc} = {sa, sb, sc} +  1;
            //$display($time, "  {sa, sb, sc}: %d",{sa, sb, sc});
            //$display($time, "  {sa, sb, sc}: %8d;\t sz1=%5d\t sz2=%5d sz3=%5d",{sa, sb, sc}, sz1, sz2, sz3);
            //$monitor($time, "  {sa, sb, sc}: %d",{sa, sb, sc});
            //$monitor($time, "  {sa, sb, sc}: %8d;\t sz1=%5d\t sz2=%5d sz3=%5d",{sa, sb, sc}, sz1, sz2, sz3);            
       end        
      #1000 $finish(1);
   end
endmodule
