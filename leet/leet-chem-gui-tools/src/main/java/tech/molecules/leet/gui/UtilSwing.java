package tech.molecules.leet.gui;

import javax.swing.*;
import java.awt.*;
import java.util.function.Supplier;

public class UtilSwing {

    public static class PanelAsFrameProvider implements Supplier<JPanel> {
        private JComponent parent;
        private int w,h;

        public PanelAsFrameProvider(JComponent parent, int w, int h) {
            this.parent = parent;
            this.w = w; this.h = h;
        }

        @Override
        public JPanel get() {
            JFrame fi = new JFrame();
            fi.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
            fi.getContentPane().setLayout(new BorderLayout());
            JPanel pi = new JPanel();
            fi.getContentPane().add(pi,BorderLayout.CENTER);
            fi.setSize(w,h);
            fi.setLocationRelativeTo(parent);
            fi.setVisible(true);
            return pi;
        }
    }

}
