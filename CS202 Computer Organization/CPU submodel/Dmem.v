`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2021/05/05 12:17:26
// Design Name: 
// Module Name: dmemory32
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


module Dmem(clock,address,Memwrite,write_data,read_data);

    input               clock; // 时钟信号
    input   [31:0]      address; // read/write memory address
    input               Memwrite; // 决定 read memory 还是 write memory
    input   [31:0]      write_data; // 写数据
    output  [31:0]      read_data; // 读数据


    // 产生一个时钟信号，它是时钟符号的反向时钟
    wire clk;
    assign clk = !clock;

    RAM ram (
        .clka(clk), // input wire clka
        .wea(Memwrite), // input wire [0 : 0] wea
        .addra(address[15:2]), // input wire [13 : 0] addra
        .dina(write_data), // input wire [31 : 0] dina
        .douta(read_data) // output wire [31 : 0] douta
    );

endmodule