import streamlit as st
import json


def display_images(images, num_columns=2):
    cols = st.columns(num_columns)
    for idx, image_data in enumerate(images):
        with cols[idx % num_columns]:
            st.image(image_data["path"], use_column_width=True, caption=image_data["caption"])


# Function to save comment to a file
def save_comment(blog_title, name, contact, comment):
    # Define the file path for comments
    file_path = f"Comments/{blog_title} comments.txt"
    # Open the file in append mode
    with open(file_path, "a") as f:
        # Write the comment information to the file
        f.write(f"Blog: {blog_title}\n")
        f.write(f"Name: {name}\n")
        f.write(f"Contact: {contact}\n")
        f.write(f"Comment: {comment}\n")
        f.write("-" * 50 + "\n")  # Separator


def display_blogs(blogs):
    for blog in blogs:
        st.subheader(blog["title"])
        st.write(blog["excerpt"])

        if "image" in blog and blog["image"]:  # Display image if present
            st.image(blog["image"], use_column_width=True)
        st.write(f"*Date*: {blog['date']}")

        # Use st.expander to create a collapsible section for the full content
        with st.expander("Read more"):
            st.write(blog["body"])

            # Comment form
            st.subheader("Leave a Comment")
            name = st.text_input("Your Name (optional)", key=f"{blog['title']} 1")
            contact = st.text_input("Your Contact (optional)", key=f"{blog['title']} 2")
            comment = st.text_area("Your Comment", height=100, key=f"{blog['title']} 3")

            if st.button("Submit Comment", key=f"{blog['title']} 4"):
                if comment:
                    save_comment(blog['title'], name, contact, comment)
                    st.success("Thank you for your comment!")
                else:
                    st.error("Comment cannot be empty!")


def load_json_data(json_file, key):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data[key]


# Sidebar for navigation
page = st.sidebar.radio("Go to", ["Gallery", "About", "Blog"])

# Gallery Page
if page == "Gallery":
    images_data = load_json_data("https://github.com/Dominic-Arthur/phd/blob/main/Website/images.json/", "images")
    st.title("Welcome to the Gallery")
    display_images(images_data, num_columns=2)

# About Page
elif page == "About":
    st.title("About Me")
    st.write("""
    Welcome to the About page.

    I'm [Your Name], a passionate photographer based in [Location].

    I specialise in [Type of Photography], and my work is inspired by [Your Inspirations]. 
    """)

# Blog Page
elif page == "Blog":
    # Load blog data
    blog_data = load_json_data("https://github.com/Dominic-Arthur/phd/blob/main/Website/blogs.json/", "blogs")

    # Blog Page
    st.title("Blog")
    st.write("Welcome to the blog. Here are some of my recent posts.")

    search_query = st.text_input("Search blog posts:")

    # Filtering functionality
    tags = set(tag for blog in blog_data for tag in blog.get("tags", []))
    selected_tag = st.multiselect("Filter by tag:", options=list(tags), default=[])

    # Filter blogs based on search query and selected tags
    filtered_blogs = [blog for blog in blog_data if
                      (search_query.lower() in blog['title'].lower() or
                       search_query.lower() in blog['excerpt'].lower()) and
                      (not selected_tag or any(tag in blog.get('tags', []) for tag in selected_tag))]

    # Display the filtered blogs in a grid

    display_blogs(filtered_blogs)

# Footer
st.write("---")

footer_container = st.container()
with footer_container:
    col1, col2 = st.columns([3, 1])  # Create three columns with different widths

    with col1:
        st.markdown(
            """
            <div style="display: flex; justify-content: space-around;">
                <p>© 2024 Dominic Arthur. All rights reserved.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
             <div style="display: flex; justify-content: space-around;">
                <a href="mailto:arthurdominic04@gmail.com" target="_blank">
                    <img src="https://img.icons8.com/ios-filled/50/000000/gmail.png" 
                    alt="Gmail" width="15" height="15"/>
                </a>
                <a href="https://twitter.com/ydnkka/" target="_blank">
                    <img src="https://img.icons8.com/ios-filled/50/000000/x.png" 
                    alt="Twitter" width="15" height="15"/>
                </a>
                <a href="https://instagram.com/ydnkka/" target="_blank">
                    <img src="https://img.icons8.com/ios-filled/50/000000/instagram-new.png" 
                    alt="Instagram" width="15" height="15"/>
                </a>
                <a href="https://linkedin.com/in/dominic-arthur/" target="_blank">
                    <img src="https://img.icons8.com/ios-filled/50/000000/linkedin.png" 
                    alt="LinkedIn" width="15" height="15"/>
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
