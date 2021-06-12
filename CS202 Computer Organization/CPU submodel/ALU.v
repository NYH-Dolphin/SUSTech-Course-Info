`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2021/05/08 17:56:30
// Design Name: 
// Module Name: Executs32
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


module ALU(Read_data_1,Read_data_2,Imme_extend,Function_opcode,opcode,Shamt,PC_plus_4,ALUOp,ALUSrc,I_format,Sftmd,Jr,Zero,ALU_Result,Addr_Result);
    // Decoder 数据
    input[31:0] Read_data_1;    // A 输入
    input[31:0] Read_data_2;    // B1 输入
    input[31:0] Imme_extend;    // B2 输入

    // ifetch 数据
    input[5:0] Function_opcode; //  instructions[5:0]
    input[5:0] opcode;          //  instruction[31:26]
    input[4:0] Shamt;           //  instruction[10:6], the amount of shift bits
    input[31:0] PC_plus_4;      //  pc+4

    // controller 数据
    input[1:0] ALUOp;
    // 如果指令是 R-type 或 I_format = 1, ALUOp = 2'b10;
    // 如果指令是 beq 或 bne, ALUOp = 2'b01;
    // 如果指令是 lw 或 sw, ALUOP = 2'b00;
     //{ (R_format || I_format) , (Branch || nBranch) }
    input ALUSrc;       // 1-第二个数是立即数（除了beq、bne）
    input I_format;     // 1-指令是 I-类型（除了beq、bne、lw、sw）
    input Sftmd;        // 1-shift 指令
    input Jr;           // 1-指令是 jr     0-表示不是 jr

    // 输出
    output wire         Zero;                // 1-ALU 结果是0  0-其它情况
    output reg          [31:0]  ALU_Result;          // ALU计算结果
    output wire         [31:0]  Addr_Result;         // 计算的指令地址

    reg     [31:0] ALU_output_mux;  // 算术或逻辑计算的结果 
    assign Zero = (ALU_output_mux[31:0] == 32'h00000000) ? 1'b1 : 1'b0;

    // 决定两个输入到底是哪个
    wire  signed [31:0] Ainput,Binput;   // 两个计算的操作数
    assign Ainput = Read_data_1;
    assign Binput = (ALUSrc == 0) ? Read_data_2 : Imme_extend[31:0];
    
    // 跳转的分支地址
    wire    [32:0] Branch_Addr;     // 该指令的计算地址，Addr_Result是 Branch_Addr[31:0]
    assign Branch_Addr = PC_plus_4[31:2] +  Imme_extend[31:0]; 
    assign Addr_Result  = Branch_Addr[31:0]; 

    // 决定 Ext_code 
    wire    [4:0] Exe_code;         // 用于生成 ALU_ctrl
    assign Exe_code = (I_format==0) ? Function_opcode : { 3'b000 , opcode[2:0]};
    // (I_format==0) ? Function_opcode : { 3'b000, Opcode[2:0] };

    // 决定 ALU_ctl
    wire    [2:0] ALU_ctl;          // 直接影响ALU操作的控制信号
    assign ALU_ctl[0] = (Exe_code[0] | Exe_code[3]) & ALUOp[1];
    assign ALU_ctl[1] = ((!Exe_code[2]) | (!ALUOp[1]));
    assign ALU_ctl[2] = (Exe_code[1] & ALUOp[1]) | ALUOp[0];

    // ALU 的计算结果
    always @(ALU_ctl or Ainput or Binput) begin
        case(ALU_ctl)
            3'b000: ALU_output_mux = Ainput & Binput;
            3'b001: ALU_output_mux = Ainput | Binput;
            3'b010: ALU_output_mux = Ainput + Binput ;
            3'b011: ALU_output_mux = Ainput + Binput;
            3'b100: ALU_output_mux = Ainput ^ Binput;
            3'b101: ALU_output_mux = ~(Ainput | Binput);
            3'b110: ALU_output_mux = Ainput - Binput;
            3'b111: ALU_output_mux = Ainput - Binput;
            default: ALU_output_mux <= 32'h0000_0000;
        endcase        
    end

    // 移位信号
    reg     [31:0] Shift_Result;    // 移位操作的结果
    wire    [2:0] Sftm;             // 识别移位指令的类型，等于 Function_opcode[2:0];
    assign Sftm = Function_opcode[2:0]; // 移位操作的编码
    always @(*) begin
        if(Sftmd) begin
          case (Sftm[2:0])
            3'b000: Shift_Result <= Binput << Shamt; // sll
            3'b010: Shift_Result <= Binput >> Shamt; // srl
            3'b100: Shift_Result <= Binput << Ainput; // sllv
            3'b110: Shift_Result <= Binput >> Ainput; // srlv
            3'b011: Shift_Result <= Binput >>> Shamt; // sra
            3'b111: Shift_Result <= Binput >>> Ainput; // srav
            default:Shift_Result <= Binput;
          endcase
        end
        else begin
          Shift_Result <= Binput;
        end
    end

    // ALU_Result
    always @(*) begin
        //set type operation (slt, slti, sltu, sltiu)
        if(((ALU_ctl==3'b111) && (Exe_code[3]==1))||((ALU_ctl[2:1]==2'b11) && (I_format==1))) begin
            ALU_Result <= (Ainput-Binput<0)?1:0;
        end
        // lui
        else if((ALU_ctl==3'b101) && (I_format==1)) begin
            ALU_Result[31:0] <= {Binput[15:0],{16{1'b0}}};
        end
        // shift
        else if(Sftmd == 1'b1) begin
            ALU_Result <= Shift_Result;
        end
        // ALU 计算结果
        else begin
            ALU_Result <= ALU_output_mux[31:0];
        end
        
    end


endmodule