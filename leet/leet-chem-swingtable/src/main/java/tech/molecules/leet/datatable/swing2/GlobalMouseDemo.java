package tech.molecules.leet.datatable.swing2;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.Arrays;

public class GlobalMouseDemo extends JFrame {

    public GlobalMouseDemo() {
        setTitle("Global Mouse Events Demo");
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setSize(400, 400);
        setLayout(new GridLayout(2, 2));

        // Create 4 JPanels with individual mouse event and mouse motion listeners
        for (int i = 0; i < 4; i++) {
            JPanel panel = new JPanel();
            panel.setBackground(new Color((int) (Math.random() * 0xFFFFFF)));
            panel.addMouseListener(new MouseAdapter() {
                @Override
                public void mouseClicked(MouseEvent e) {
                    System.out.println("Component Mouse Clicked: " + e.getPoint());
                }
            });
            panel.addMouseMotionListener(new MouseMotionAdapter() {
                @Override
                public void mouseMoved(MouseEvent e) {
                    System.out.println("Component Mouse Moved: " + e.getPoint());
                }
            });
            add(panel);
        }

        // Global mouse event and mouse motion listener
        AWTEventListener globalMouseListener = new AWTEventListener() {
            @Override
            public void eventDispatched(AWTEvent event) {
                if (event instanceof MouseEvent) {
                    MouseEvent mouseEvent = (MouseEvent) event;

                    // Filter events to the specific JPanels within the JFrame
                    if (Arrays.asList(getContentPane().getComponents()).contains(mouseEvent.getComponent())) {
                        if (mouseEvent.getID() == MouseEvent.MOUSE_CLICKED) {
                            System.out.println("Global Mouse Clicked: " + mouseEvent.getPoint());
                        } else if (mouseEvent.getID() == MouseEvent.MOUSE_MOVED) {
                            System.out.println("Global Mouse Moved: " + mouseEvent.getPoint());
                        }
                    }
                }
            }
        };

        // Register the global mouse event and mouse motion listener
        long eventMask = AWTEvent.MOUSE_EVENT_MASK | AWTEvent.MOUSE_MOTION_EVENT_MASK;
        Toolkit.getDefaultToolkit().addAWTEventListener(globalMouseListener, eventMask);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            GlobalMouseDemo demo = new GlobalMouseDemo();
            demo.setVisible(true);
        });
    }
}
