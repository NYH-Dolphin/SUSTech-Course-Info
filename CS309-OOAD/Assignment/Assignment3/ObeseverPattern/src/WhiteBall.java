import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;

public class WhiteBall extends Ball implements Subject<Ball>{
    public WhiteBall(Color color, int xSpeed, int ySpeed, int ballSize) {
        super(color, xSpeed, ySpeed, ballSize);
    }
    private List<Ball> observers = new ArrayList<>();

    @Override
    public void update(char keyChar) {
        if (MainPanel.getGameStatus() == MainPanel.GameStatus.START) {
            switch (keyChar) {
                case 'a':
                    this.setXSpeed(-8);
                    break;
                case 'd':
                    this.setXSpeed(8);
                    break;
                case 'w':
                    this.setYSpeed(-8);
                    break;
                case 's':
                    this.setYSpeed(8);
                    break;
            }
        }
    }


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    @Override
    public void update(Ball ball) {
        throw new NotImplementedException(); // 白球不需要实现
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
    public void move(){
        super.move();
        notifyObservers();
    }

    @Override
    public void notifyObservers(char keyChar) {
    }

    @Override
    public void notifyObservers() {
        observers.forEach(ball -> ball.update(this));
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
}
