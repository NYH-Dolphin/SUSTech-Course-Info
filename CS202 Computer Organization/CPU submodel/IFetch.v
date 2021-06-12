`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2021/05/08 21:28:00
// Design Name: 
// Module Name: Ifetc32
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


module IFetch(clock,reset,Addr_result,Zero,Read_data_1,Branch,nBranch,Jmp,Jal,Jr,Instruction,branch_base_addr,link_addr,pco);

    input clock,reset;

    // from ALU
    input[31:0] Addr_result; // 从ALU计算出的地址
    input       Zero; // 1-ALU Reuslt = 0

    // from Decoder
    input[31:0] Read_data_1; // jr指令所使用的指令地址

    // from controller
    input       Branch;     // 1-beq
    input       nBranch;    // 1-bne
    input       Jmp;        // 1-j
    input       Jal;        // 1-jal
    input       Jr;         // 1-jr

    output  [31:0]  Instruction;        // 从该模块中获取的指令
    output  [31:0]  branch_base_addr;   // (pc+4)到ALU，这是由分支类型指令使用
    output  reg [31:0]  link_addr;          // (pc+4)到jal指令使用的解码器
    output  [31:0]  pco;

    reg [31:0] PC, Next_PC;


    always @(*) begin
        // jr
        if(Jr == 1'b1) begin
          Next_PC <= Read_data_1 * 4;
        end
        // beq, bne
        else if(((Branch == 1'b1) && (Zero == 1'b1)) || ((nBranch == 1'b1) && (Zero == 1'b0))) begin
            Next_PC <= Addr_result * 4;
        end
        // PC + 4
        else begin
            Next_PC <= PC + 4;
        end
    end

    always @(negedge clock) begin
        if(reset == 1'b1) begin
            PC <= 32'h0000_0000;
        end
        else begin
          // j 或 jal
          if((Jmp == 1'b1)|| (Jal == 1'b1)) begin
            PC <= {4'b0000, Instruction[25:0],2'b00};
          end
          // 其它的正常更新
          else begin
            PC <= Next_PC;
          end
        end
    end

    assign pco = PC;
    assign branch_base_addr = PC + 4;

    prgrom myRAM(
    .clka(clock), // input wire clka
    .addra(PC[15:2]), // input wire [13 : 0] addra
    .douta(Instruction) // output wire [31 : 0] douta
    );

    
    always @(posedge Jmp, posedge Jal) begin
        link_addr <= branch_base_addr / 4;
    end


endmodule