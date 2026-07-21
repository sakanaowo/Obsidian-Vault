# What are you doing in here? 🤨

<?xml version="1.0" encoding="utf-8"?>
<resources>

    <!-- Base calculator button -->
    <style name="CalculatorButtonBase">

        <item name="android:layout_height">
            @dimen/calculator_button_height
        </item>

        <item name="android:textColor">
            @color/calculator_text_primary
        </item>

        <item name="android:fontFamily">
            sans
        </item>

        <item name="android:textAllCaps">
            false
        </item>

        <item name="android:gravity">
            center
        </item>

        <item name="android:padding">
            0dp
        </item>

        <!-- Prevent theme tint from overriding drawable -->
        <item name="android:backgroundTint">
            @null
        </item>

    </style>


    <!-- Number -->
    <style
        name="CalculatorButtonNumber"
        parent="CalculatorButtonBase">

        <item name="android:background">
            @drawable/bg_button_number
        </item>

        <item name="android:textSize">
            @dimen/text_button_number
        </item>

    </style>


    <!-- Function -->
    <style
        name="CalculatorButtonFunction"
        parent="CalculatorButtonBase">

        <item name="android:background">
            @drawable/bg_button_function
        </item>

        <item name="android:textSize">
            @dimen/text_button_function
        </item>

    </style>


    <!-- Operator -->
    <style
        name="CalculatorButtonOperator"
        parent="CalculatorButtonBase">

        <item name="android:background">
            @drawable/bg_button_operator
        </item>

        <item name="android:textSize">
            @dimen/text_button_operator
        </item>

    </style>

</resources>