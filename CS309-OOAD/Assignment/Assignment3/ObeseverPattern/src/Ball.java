import javax.swing.*;
import java.awt.*;

public abstract class Ball extends JComponent{
    private Color color;
    private int x, y;
    private int xSpeed, ySpeed;
    private final int ballSize;
    private boolean visible;
    public static final int TOTAL_NUM = 15;
    private static int count = 0;

    public Ball(Color color, int xSpeed, int ySpeed, int ballSize) {
        this.color = color;
        this.xSpeed = xSpeed;
        this.ySpeed = ySpeed;
        this.ballSize = ballSize;

        this.visible = true;
        this.x = (int) (Math.random() * 580);
        this.y = (int) (Math.random() * 580);
        count++;

        this.setSize(ballSize, ballSize);
    }

    public static int getCount() {
        return count;
    }

    public static void setCount(int count) {
        Ball.count = count;
    }

    public void setColor(Color newColor) {
        this.color = newColor;
    }

    public Color getColor() {
        return this.color;
    }

    public int getX() {
        return x;
    }

    public void setX(int x) {
        this.x = x;
    }

    public int getY() {
        return y;
    }

    public void setY(int y) {
        this.y = y;
    }

    public int getYSpeed() {
        return ySpeed;
    }

    public void setYSpeed(int ySpeed) {
        this.ySpeed = ySpeed;
    }

    public int getXSpeed() {
        return xSpeed;
    }

    public void setXSpeed(int xSpeed) {
        this.xSpeed = xSpeed;
    }

    public int getBallSize() {
        return ballSize;
    }

    public boolean isVisible() {
        return visible;
    }

    public void setVisible(boolean visible) {
        this.visible = visible;
    }

    public void draw(Graphics g) {
        if (isVisible()) {
            if (color == Color.WHITE) {
                g.setColor(Color.BLACK);
                g.drawOval(0, 0, this.getBallSize(), this.getBallSize());
            } else {
                g.setColor(this.getColor());
                g.fillOval(0, 0, this.getBallSize(), this.getBallSize());
            }
        }

    }

    public void move() {
        int newX = this.getX() + this.getXSpeed();
        int newY = this.getY() + this.getYSpeed();

        this.setX(newX);
        this.setY(newY);

        if (newX <= 0) {
            this.setXSpeed(Math.abs(getXSpeed()));
        } else if (newX >= 600 - this.getBallSize()) {
            this.setXSpeed(-1 * Math.abs(getXSpeed()));
        }

        if (newY <= 0) {
            this.setYSpeed(Math.abs(getYSpeed()));
        } else if (newY > 600 - this.getBallSize()) {
            this.setYSpeed(-1 * Math.abs(getYSpeed()));
        }
    }

    public int convertToCentralX() {
        return this.getX() + this.getWidth() / 2;
    }

    public int convertToCentralY() {
        return this.getY() + this.getHeight() / 2;
    }

    public boolean isIntersect(Ball b) {
        int dis = (this.getWidth() + b.getWidth()) / 2;
        double diffXPow = Math.pow(this.convertToCentralX() - b.convertToCentralX(), 2);
        double diffYPow = Math.pow(this.convertToCentralY() - b.convertToCentralY(), 2);
        return diffXPow + diffYPow < dis * dis;
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        draw(g);
    }


    public abstract void update(char keyChar);

    public abstract void update(Ball ball);
}

