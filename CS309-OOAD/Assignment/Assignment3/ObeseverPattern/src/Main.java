import javax.swing.*;

public class Main {
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            MainFrame mainGameFrame = new MainFrame();
            mainGameFrame.setVisible(true);
        });
    }
}
