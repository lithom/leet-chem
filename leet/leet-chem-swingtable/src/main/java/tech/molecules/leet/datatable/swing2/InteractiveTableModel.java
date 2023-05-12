package tech.molecules.leet.datatable.swing2;

import com.actelion.research.chem.Depictor2D;
import com.actelion.research.chem.IDCodeParser;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.gui.JStructureView;
import org.apache.commons.lang3.tuple.Pair;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.util.Map;
import java.util.Random;

public class InteractiveTableModel {

    public int getRows() {
        return 1000;
    }

    public int getColumns() {
        return 12;
    }

    public int getRow_Y(int row) {
        return row * 80;
    }

    public int getCol_X(int col) {
        return col * 120;
    }

    public int getCol_W(int col) {
        return getCol_X(col+1)-getCol_X(col);
    }

    public int getRow_H(int row) {
        return getCol_X(row+1)-getCol_X(row);
    }


    private StereoMolecule mi = new StereoMolecule();
    {
        IDCodeParser ip = new IDCodeParser();
        ip.parse(mi,"fewI@@DLStFQQQIIZIIQSIIDRoYjMoVjjdH@`FJB@@");
    }

    private Map<Pair<Integer,Integer>,JComponent> components;

    public JComponent createComponent(int row, int col) {

        JPanel pi = new JPanel();
        pi.setSize(120, 80);
        pi.setLayout(new GridLayout(2, 2));

        if( col < 2) {
            StereoMolecule ma = new StereoMolecule();
            IDCodeParser ip = new IDCodeParser();
            ip.parse(ma,"fewI@@DLStFQQQIIZIIQSIIDRoYjMoVjjdH@`FJB@@");
            JStructureView jsv = new JStructureView(ma);
            return jsv;
        }

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
            JLabel li = new JLabel(rg.generateRandomString(16));
            panel.add(li);
            pi.add(panel);
        }
        return pi;
    }

    RandomStringGenerator rg = new RandomStringGenerator();

    public static class RandomStringGenerator {

        private static final String ALPHABET   = "abcdefg";
        private static final String ALPHABET_B = "abcdefghijklm+*ç%&/()=?nopqrstuvöä¨$¨üwx§y<>zABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
        private static final String ALPHABET_A = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
        //private static final int LENGTH = 10;

        public String generateRandomString(int length) {
            Random random = new Random();
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < length; i++) {
                int index = random.nextInt(ALPHABET.length());
                builder.append(ALPHABET.charAt(index));
            }
            return builder.toString();
        }
    }

}
