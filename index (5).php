<!DOCTYPE html>
<html>
<head>
    <title>PHP Web App</title>
</head>
<body>
	    <h1>Hello, PHP!</h1>
	
	    <p>This is a PHP web app that sends a POST request.</p>

	    <?php
	    if ($_SERVER['REQUEST_METHOD'] === 'POST') {
		// Retrieve the question from the form
		$question = $_POST['question'];
		$document = $_POST['document'];

		// Set the request URL
		$url = 'http://10.1.0.4:8000/qanda';

		// Set the request data
		$data = array(
		    'question' => $question,
	            'document' =>$document,
		);

		// Initialize cURL session
		$ch = curl_init();

		// Set cURL options
		curl_setopt($ch, CURLOPT_URL, $url);
		curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
		curl_setopt($ch, CURLOPT_POSTFIELDS, http_build_query($data));

		// Execute the request
		$response = curl_exec($ch);

		// Check for errors
		if (curl_errno($ch)) {
		    echo 'Error: ' . curl_error($ch);
		} else {
		    // Output the response
		    echo $response;
		}

		// Close cURL session
		curl_close($ch);
	    }
	    ?>

	    <form action="" method="POST">
		<label for="question">Question:</label>
		<input type="text" name="question" id="question" required><br>
		<input type="radio" name="document" value="aws_general_vectorstore.pkl">aws_general_vectorstore.pkl<br>
                <input type="radio" name="document" value="cisco-catalyst-9164-series-access-points-ds_vectorstore.pkl">cisco-catalyst-9164-series-access-points-ds_vectorstore.pkl<br>
                <input type="radio" name="document" value="cisco-End-of-Sale and End-of-Life Announcement for the Cisco Aironet 4800 Series Access Points - Cisco_vectorstore.pkl">cisco-End-of-Sale and End-of-Life Announcement for the Cisco Aironet 4800 Series Access Points - Cisco_vectorstore.pkl<br>
                <input type="radio" name="document" value="cisco-vectorstore.pkl">cisco-vectorstore.pkl<br>
                <input type="radio" name="document" value="codecatalyst_action_dg_vectorstore.pkl">codecatalyst_action_dg_vectorstore.pkl<br>
                <input type="radio" name="document" value="codecatalyst_api_vectorstore.pkl">codecatalyst_api_vectorstore.pkl<br>
                <input type="radio" name="document" value="codecatalyst_ug_vectorstore.pkl">codecatalyst_ug_vectorstore.pkl<br>
                #<input type="radio" name="document" value="merged_vectorstore.pkl">merged_vectorstore.pkl<br>
                <input type="radio" name="document" value="sagemaker_api_vectorstore.pkl">sagemaker_api_vectorstore.pkl<br>
                <input type="radio" name="document" value="sagemaker_dg_vectorstore.pkl">sagemaker_dg_vectorstore.pkl<br>
                

		<button type="submit">Submit</button>
	    </form>
</body>
</html>
