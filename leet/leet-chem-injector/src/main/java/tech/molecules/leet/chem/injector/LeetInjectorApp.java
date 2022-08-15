package tech.molecules.leet.chem.injector;

import com.formdev.flatlaf.FlatLightLaf;
import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.chem.dataimport.FragmentDBCreator;
import tech.molecules.leet.chem.injector.gui.JInjectorMainPanel;

import javax.swing.*;
import java.awt.*;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Hello world!
 *
 */
public class LeetInjectorApp
{
    public static void main( String[] args )
    {

        System.out.println( "Hello World!" );

        List<Pair<String,Integer>> fragdb = FragmentDBCreator.loadFragments2();
        //Injector injector = new Injector( fragdb.stream().map( pi -> pi.getLeft() ).collect(Collectors.toList()) );
        Injector.initInjector( fragdb.stream().map( pi -> pi.getLeft() ).collect(Collectors.toList()) );

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
