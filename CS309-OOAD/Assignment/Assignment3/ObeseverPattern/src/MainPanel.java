import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MainPanel extends JPanel implements KeyListener, Subject<Ball> {
    Random random = new Random();
    Timer t;
    private List<Ball> paintingBallList = new ArrayList<>();
    private static GameStatus gameStatus;
    private int score;
    private Ball whiteBall;

    public static GameStatus getGameStatus() {
        return gameStatus;
    }

    public List<Ball> getPaintingBallList() {
        return paintingBallList;
    }

    private List<Ball> observers = new ArrayList<>();
    public MainPanel() {
        super();
        setLayout(null);
        setSize(590, 590);
        setFocusable(true);
        this.addKeyListener(this);
        t = new Timer(50, e -> moveBalls());
        restartGame();
    }

    @Override
    public void registerObserver(Ball ball) {
        observers.add(ball);
    }

    @Override
    public void removeObserver(Ball ball) {
        observers.remove(ball);
    }

    @Override
    public void notifyObservers(char keyChar) {
        observers.forEach(ball -> ball.update(keyChar));
    }

    @Override
    public void notifyObservers() {
        throw new NotImplementedException();
    }
    public void startGame() {
        gameStatus = GameStatus.START;
        this.whiteBall.setVisible(true);
        this.paintingBallList.forEach(b -> b.setVisible(false));
    }

    public void stopGame() {
        gameStatus = GameStatus.STOP;
        this.t.stop();
        paintingBallList.forEach(b -> {
            if (b.isVisible()) {
                if (b.getColor() == Color.RED) {
                    scoreIncrement(80);
                } else if (b.getColor() == Color.BLUE) {
                    scoreIncrement(-80);
                }
            }
        });
        repaint();
    }

    public void restartGame() {
        gameStatus = GameStatus.PREPARING;
        if (paintingBallList.size() > 0) {
            paintingBallList.forEach(this::remove);
        }
        this.paintingBallList = new ArrayList<>();
        Ball.setCount(0);
        this.score = 100;
        if (this.whiteBall != null)
            this.whiteBall.setVisible(false);

        this.t.start();
        repaint();
    }

    public void setWhiteBall(Ball whiteBall) {
        this.whiteBall = whiteBall;
        this.whiteBall.setVisible(false);
        add(whiteBall);
    }

    public void moveBalls() {
        paintingBallList.forEach(Ball::move);
        if (gameStatus == GameStatus.START) {
            score--;
            whiteBall.move();
            paintingBallList.forEach(b -> {
                b.setVisible(b.isIntersect(whiteBall));
            });
        }
        repaint();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        g.setFont(new Font("Arial", Font.PLAIN, 30));
        g.setColor(Color.BLACK);
        g.drawString("Score: " + score, 20, 40);

        if (gameStatus == GameStatus.START) {
            this.setBackground(Color.WHITE);
        }

        if (gameStatus == GameStatus.STOP) {
            g.setColor(Color.BLACK);
            g.setFont(new Font("Arial", Font.BOLD, 45));
            g.drawString("Game Over!", 200, 200);
            g.setFont(new Font("", Font.BOLD, 40));
            g.drawString("Your score is " + score, 190, 280);
        }
    }

    public void scoreIncrement(int increment) {
        this.score += increment;
    }

    public void addBallToPanel(Ball ball) {
        paintingBallList.add(ball);
        this.add(ball);
    }

    @Override
    public void keyTyped(KeyEvent e) {

    }

    @Override
    public void keyPressed(KeyEvent e) {
        char keyChar = e.getKeyChar();
        System.out.println("Press: " + keyChar);
        notifyObservers(keyChar);
    }

    @Override
    public void keyReleased(KeyEvent e) {

    }

    enum GameStatus {PREPARING, START, STOP}


}
