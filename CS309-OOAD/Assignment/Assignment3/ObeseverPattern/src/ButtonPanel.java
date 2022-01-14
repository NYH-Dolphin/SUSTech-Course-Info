import javax.swing.*;
import java.awt.*;

public class ButtonPanel extends JPanel {
    private final JButton addRedBtn;
    private final JButton addBlueBtn;
    private final JButton startBtn;
    private final JButton stopBtn;
    private final JButton restartBtn;
    private final JLabel redCountLabel;
    private final JLabel blueCountLabel;

    public ButtonPanel() {
        setSize(100, 590);
        setLayout(null);
        setBackground(Color.lightGray);


        //从底向上分别是background>foreground>border layer
        addRedBtn = new JButton("RED");
        addRedBtn.setFont(new Font("", Font.BOLD, 15));
        addRedBtn.setSize(80, 80);
        addRedBtn.setLocation(10, 20);
        addRedBtn.setVisible(true);

        redCountLabel = new JLabel("Red Count: 0");
        redCountLabel.setSize(90, 30);
        redCountLabel.setLocation(10, 105);
        redCountLabel.setFont(new Font("", Font.BOLD, 12));
        redCountLabel.setVisible(true);
        redCountLabel.setForeground(Color.RED);

        addBlueBtn = new JButton("BLUE");
        addBlueBtn.setFont(new Font("", Font.BOLD, 15));
        addBlueBtn.setSize(80, 80);
        addBlueBtn.setLocation(10, 150);
        addBlueBtn.setVisible(true);

        blueCountLabel = new JLabel("Blue Count: 0");
        blueCountLabel.setSize(90, 30);
        blueCountLabel.setLocation(10, 235);
        blueCountLabel.setFont(new Font("", Font.BOLD, 12));
        blueCountLabel.setVisible(true);
        blueCountLabel.setForeground(Color.BLUE);


        startBtn = new JButton("START");
        startBtn.setFont(new Font("", Font.BOLD, 12));
        startBtn.setSize(80, 80);
        startBtn.setLocation(10, 280);
        startBtn.setVisible(true);

        stopBtn = new JButton("STOP");
        stopBtn.setFont(new Font("", Font.BOLD, 12));
        stopBtn.setSize(80, 80);
        stopBtn.setLocation(10, 390);
        stopBtn.setVisible(true);

        restartBtn = new JButton("RESTART");
        restartBtn.setFont(new Font("", Font.BOLD, 12));
        restartBtn.setSize(80, 80);
        restartBtn.setLocation(10, 500);
        restartBtn.setVisible(true);


        this.add(addRedBtn);
        this.add(addBlueBtn);
        this.add(startBtn);
        this.add(stopBtn);
        this.add(restartBtn);
        this.add(redCountLabel);
        this.add(blueCountLabel);
    }

    public JButton getAddRedBtn() {
        return addRedBtn;
    }

    public JButton getAddBlueBtn() {
        return addBlueBtn;
    }

    public JButton getStartBtn() {
        return startBtn;
    }

    public JButton getStopBtn() {
        return stopBtn;
    }

    public JButton getRestartBtn() {
        return restartBtn;
    }

    public JLabel getRedCountLabel() {
        return redCountLabel;
    }

    public JLabel getBlueCountLabel() {
        return blueCountLabel;
    }
}
