import javax.swing.*;
import java.awt.*;

public class MainFrame extends JFrame {
    ButtonPanel buttonPanel;
    MainPanel mainPanel;
    int redCount;
    int blueCount;

    public MainFrame() {
        setTitle("Observer Pattern 2021");
        setSize(720, 630);
        setBackground(Color.gray);
        setLocationRelativeTo(null);
        setLayout(null);
        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        mainPanel = new MainPanel();
        buttonPanel = new ButtonPanel();

        mainPanel.setLocation(5, 5);
        this.add(mainPanel);

        buttonPanel.setLocation(600, 5);
        this.add(buttonPanel);


        JButton red = buttonPanel.getAddRedBtn();
        JButton blue = buttonPanel.getAddBlueBtn();
        JButton start = buttonPanel.getStartBtn();
        JButton stop = buttonPanel.getStopBtn();
        JButton restart = buttonPanel.getRestartBtn();


        stop.setEnabled(false);
        restart.setEnabled(false);

        WhiteBall whiteBall = new WhiteBall(Color.WHITE, 0, 0, 200);
        mainPanel.setWhiteBall(whiteBall);
        mainPanel.registerObserver(whiteBall);
        Ball.setCount(0);


        red.addActionListener(l -> {
            if (Ball.getCount() < Ball.TOTAL_NUM) {
                RedBall redBall = new RedBall(Color.RED, 3, 2, 60);
                mainPanel.addBallToPanel(redBall);
                mainPanel.registerObserver(redBall);
                mainPanel.scoreIncrement(-10);
                redCount++;
                buttonPanel.getRedCountLabel().setText("RED: " + redCount);

                whiteBall.registerObserver(redBall);
            }
        });

        blue.addActionListener(l -> {
            if (Ball.getCount() < Ball.TOTAL_NUM) {
                BlueBall blueBall = new BlueBall(Color.BLUE, 6, 4, 60);
                mainPanel.addBallToPanel(blueBall);
                mainPanel.registerObserver(blueBall);
                mainPanel.scoreIncrement(+30);
                blueCount++;
                buttonPanel.getBlueCountLabel().setText("BlUE: " + blueCount);

                whiteBall.registerObserver(blueBall);
            }
        });

        start.addActionListener(l -> {
            mainPanel.startGame();
            red.setEnabled(false);
            blue.setEnabled(false);
            start.setEnabled(false);
            stop.setEnabled(true);
            restart.setEnabled(false);
        });

        stop.addActionListener(l -> {
            mainPanel.stopGame();
            red.setEnabled(false);
            blue.setEnabled(false);
            start.setEnabled(false);
            stop.setEnabled(false);
            restart.setEnabled(true);
        });

        restart.addActionListener(l -> {
            mainPanel.getPaintingBallList().forEach(ball -> mainPanel.removeObserver(ball));
            mainPanel.restartGame();
            red.setEnabled(true);
            blue.setEnabled(true);
            start.setEnabled(true);
            stop.setEnabled(false);
            restart.setEnabled(false);
            initialCount();
        });

    }

    public ButtonPanel getButtonPanel() {
        return buttonPanel;
    }

    public MainPanel getMainPanel() {
        return mainPanel;
    }

    public void initialCount() {
        this.redCount = 0;
        this.blueCount = 0;
        buttonPanel.getRedCountLabel().setText("RED: " + redCount);
        buttonPanel.getBlueCountLabel().setText("BLUE: " + blueCount);
    }
}
