`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2021/05/08 12:42:57
// Design Name: 
// Module Name: control32
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


module Controller(Opcode,Function_opcode,Jr,Jmp,Jal,Branch,nBranch,RegDST,MemtoReg,RegWrite,MemWrite,ALUSrc,I_format,Sftmd,ALUOp);
    input[5:0] Opcode; 
    input[5:0] Function_opcode;
    
    output Jr;          // 1-指令是 jr     0-表示不是 jr
    output Jmp;         // 1-指令是 j      0-不是 j
    output Jal;         // 1-指令是 jal    0-不是 jal
    output Branch;      // 1-指令是 beq    0-不是beq
    output nBranch;     // 1-指令是 bne    0-不是bne
    output RegDST;      // 1-写寄存器的目标寄存器来自 rd 字段 [15:11]   0-写寄存器的目标寄存器来自 rt 字段 [20:16]
    output MemtoReg;    // 1-写寄存器的数据来自数据存储器               0-写寄存器的数据来自 ALU
    output RegWrite;    // 1-寄存器堆写使能有效
    output MemWrite;    // 1-写内存
    output ALUSrc;      // 1-第二个数是立即数（除了beq、bne）
    output I_format;    // 1-指令是 I-类型（除了beq、bne、lw、sw）
    output Sftmd;       // 1-shift 指令

    output[1:0] ALUOp;
    // 如果指令是 R-type 或 I_format = 1, ALUOp = 2'b10;
    // 如果指令是 beq 或 bne, ALUOp = 2'b01;
    // 如果指令是 lw 或 sw, ALUOP = 2'b00;

    // 表示指令是否为 R 型指令
    wire R_format; 
    assign R_format = (Opcode==6'b000000)? 1'b1:1'b0;

    // 表示指令是否为 lw 指令
    wire Lw; 
    assign Lw = (Opcode==6'b100011)? 1'b1:1'b0;
    wire Sw;
    assign Sw = (Opcode==6'b101011) ? 1'b1:1'b0;

    // 表示指令是否为 I-类型（除了beq、bne、lw、sw） 立即数指令
    assign I_format = (Opcode[5:3]==3'b001)? 1'b1:1'b0;

    // 决定跳转指令
    assign Jr       = ((Function_opcode==6'b001000) && (Opcode==6'b000000)) ? 1'b1:1'b0;
    assign Jmp      = (Opcode==6'b000010) ? 1'b1:1'b0;
    assign Jal      = (Opcode==6'b000011) ? 1'b1:1'b0;
    assign Branch   = (Opcode==6'b000100) ? 1'b1:1'b0;
    assign nBranch  = (Opcode==6'b000101) ? 1'b1:1'b0;

    // 决定控制信号
    assign RegDST = R_format;
    assign MemtoReg = Lw; // lw
    assign RegWrite = (R_format || Lw || Jal || I_format) && !(Jr);
    assign MemWrite = Sw; // sw
    assign ALUSrc = I_format || Lw || Sw;
    assign Sftmd = (((Function_opcode==6'b000000)||(Function_opcode==6'b000010)||(Function_opcode==6'b000011)||(Function_opcode==6'b000100)||(Function_opcode==6'b000110)||(Function_opcode==6'b000111))&& R_format)? 1'b1:1'b0;
    assign ALUOp = {(R_format || I_format),(Branch || nBranch)};


endmodule