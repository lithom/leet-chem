package tech.molecules;

import com.formdev.flatlaf.FlatLightLaf;
import tech.molecules.leet.chem.injector.gui.JInjectorMainPanel;

import javax.swing.*;
import java.awt.*;

/**
 * Hello world!
 *
 */
public class LeetInjector
{
    public static void main( String[] args )
    {

        System.out.println( "Hello World!" );

        FlatLightLaf.setup();
        try {
            UIManager.setLookAndFeel( new FlatLightLaf() );
        } catch( Exception ex ) {
            System.err.println( "Failed to initialize LaF" );
        }


        JFrame fi = new JFrame();
        fi.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);

        fi.getContentPane().setLayout(new BorderLayout());;

        JInjectorMainPanel main = new JInjectorMainPanel();
        fi.getContentPane().add(main,BorderLayout.CENTER);

        fi.setSize(800,600);
        fi.setVisible(true);

    }
}
